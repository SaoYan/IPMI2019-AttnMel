import os
import csv
import cv2
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as utils
import torchvision.transforms as torch_transforms
from networks import VGGGAP
from data_2017 import preprocess_data, ISIC
from utilities import *
from transforms import *

import matplotlib.pyplot as plt

'''
switch between ISIC 2016 and 2017
modify the following contents:
1. import
2. root_dir of preprocess_data
3. mean and std of transforms.Normalize
'''

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

parser = argparse.ArgumentParser(description="Attn-SKin-test")

parser.add_argument("--preprocess", action='store_true', help="run preprocess_data")

parser.add_argument("--outf", type=str, default="logs_test", help='path of log files')
parser.add_argument("--base_up_factor", type=int, default=8, help="number of epochs")

parser.add_argument("--normalize_attn", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')
parser.add_argument("--no_attention", action='store_true', help='turn off attention')

opt = parser.parse_args()

def main():
    # prepare for CAM
    features_blobs = []
    def hook_feature(module, input, output):
        feature = F.max_pool2d(output, 2, 2)
        features_blobs.append(feature.to('cpu').numpy())

    # load data
    print('\nloading the dataset ...\n')
    transform_test = torch_transforms.Compose([
         RatioCenterCrop(0.8),
         Resize((256,256)),
         CenterCrop((224,224)),
         ToTensor(),
         Normalize((0.6820, 0.5312, 0.4736), (0.0840, 0.1140, 0.1282)) # ISIC 2017
         # Normalize((0.7012, 0.5517, 0.4875), (0.0942, 0.1331, 0.1521)) # ISIC 2016
    ])
    testset = ISIC(csv_file='test.csv', shuffle=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)
    print('done')

    # load network
    print('\nloading the model ...\n')
    if not opt.no_attention:
        print('\nturn on attention ...\n')
        if opt.normalize_attn:
            print('\nuse softmax for attention map ...\n')
        else:
            print('\nuse sigmoid for attention map ...\n')
    else:
        print('\nturn off attention ...\n')

    net = VGGGAP(num_classes=2, attention=not opt.no_attention, normalize_attn=opt.normalize_attn)
    net._modules.get('conv_block5').register_forward_hook(hook_feature)
    checkpoint = torch.load('checkpoint_gap.pth')
    net.load_state_dict(checkpoint['state_dict'])
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.eval()
    print('done')

    # testing
    print('\nstart testing ...\n')
    writer = SummaryWriter(opt.outf)
    total = 0
    correct = 0
    J = 0.
    with torch.no_grad():
        with open('test_results.csv', 'wt', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for i, data in enumerate(testloader, 0):
                features_blobs.clear()
                images_test, seg_test, labels_test = data['image'], data['image_seg'], data['label']
                seg_test = seg_test[:,-1:,:,:]
                images_test, labels_test = images_test.to(device), labels_test.to(device)
                pred_test, __, __ = model.forward(images_test)
                predict = torch.argmax(pred_test, 1)
                total += labels_test.size(0)
                correct += torch.eq(predict, labels_test).sum().double().item()
                # record test predicted responses
                responses = F.softmax(pred_test, dim=1).squeeze().cpu().numpy()
                responses = [responses[i] for i in range(responses.shape[0])]
                csv_writer.writerow(responses)
                # CAM
                params = model.state_dict()['module.classify.weight']
                weights = np.squeeze(params.to('cpu').numpy())
                cam = weights[1].dot(features_blobs[0].reshape((512, 7*7)))
                cam = cam.reshape(7, 7)
                min_val, max_val = np.min(cam), np.max(cam)
                cam = (cam - min_val) / (max_val - min_val)
                mask = np.greater_equal(cam, 0.5).astype(np.float32)
                mask = np.uint8(255 * mask)
                mask = np.float32(cv2.resize(mask,(224, 224))) / 255.
                J += jaccard_similarity_coefficient(mask, np.squeeze(seg_test.numpy()))
    precision, recall, precision_mel, recall_mel = compute_mean_pecision_recall('test_results.csv')
    mAP, AUC, __ = compute_metrics('test_results.csv')
    print("\ntest result: accuracy %.2f%% \nmean precision %.2f%% mean recall %.2f%% \
            \nprecision for mel %.2f%% recall for mel %.2f%% \nmAP %.2f%% AUC %.4f\n" %
            (100*correct/total, 100*precision, 100*recall, 100*precision_mel, 100*recall_mel, 100*mAP, AUC))
    print(J / len(testloader))

if __name__ == "__main__":
    if opt.preprocess:
        preprocess_data(root_dir='../data_2017', seg_dir='Train_Lesion')
        # preprocess_data(root_dir='../data_2016')
    main()
