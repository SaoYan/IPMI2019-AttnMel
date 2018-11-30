import os
import csv
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from networks import AttnVGG
from utilities import *
from data_2017 import preprocess_data, ISIC

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
parser.add_argument("--log_images", action='store_true', help='log images')

opt = parser.parse_args()

def main():
    # load data
    print('\nloading the dataset ...\n')
    im_size = 224
    transform_test = transforms.Compose([
        transforms.CenterCrop(im_size),
        transforms.ToTensor()
    ])
    testset = ISIC(csv_file='test.csv', shuffle=False, resize=(256,256), randcrop=None, rotate=False, flip=False,
        transform=transform_test, transform_seg=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

    # ISIC 2017: mean & std of the dataset
    Mean = torch.from_numpy(np.array([0.6900, 0.5442, 0.4865]).astype(np.float32))
    Std  = torch.from_numpy(np.array([0.0810, 0.1118, 0.1265]).astype(np.float32))
    # ISIC 2016: mean & std of the dataset
    # Mean = torch.from_numpy(np.array([0.7105, 0.5646, 0.4978]).astype(np.float32))
    # Std  = torch.from_numpy(np.array([0.0910, 0.1310, 0.1510]).astype(np.float32))
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

    net = AttnVGG(num_classes=2, attention=not opt.no_attention, normalize_attn=opt.normalize_attn)
    checkpoint = torch.load('checkpoint.tar')
    net.load_state_dict(checkpoint['state_dict'])
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.eval()
    print('done')

    # testing
    print('\nstart testing ...\n')
    writer = SummaryWriter(opt.outf)
    total = 0
    correct = 0
    with torch.no_grad():
        with open('test_results.csv', 'wt', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for i, data in enumerate(testloader, 0):
                images_test, __, labels_test = data
                images_test = (images_test - Mean.view(1,3,1,1)) / Std.view(1,3,1,1)
                images_test, labels_test = images_test.to(device), labels_test.to(device)
                pred_test, __, __, __ = model.forward(images_test)
                predict = torch.argmax(pred_test, 1)
                total += labels_test.size(0)
                correct += torch.eq(predict, labels_test).sum().double().item()
                # record test predicted responses
                responses = F.softmax(pred_test, dim=1).squeeze().cpu().numpy()
                responses = [responses[i] for i in range(responses.shape[0])]
                csv_writer.writerows(responses)
                # log images
                if opt.log_images:
                    I_test = utils.make_grid(images_test, nrow=8, normalize=True, scale_each=True)
                    writer.add_image('test/image', I_test, i)
                    # accention maps
                    if not opt.no_attention:
                        __, a1, a2, a3 = model.forward(images_test)
                        if a1 is not None:
                            attn1, __ = visualize_attn(I_test, a1, up_factor=opt.base_up_factor, nrow=8)
                            writer.add_image('test/attention_map_1', attn1, i)
                        if a2 is not None:
                            attn2, __ = visualize_attn(I_test, a2, up_factor=2*opt.base_up_factor, nrow=8)
                            writer.add_image('test/attention_map_2', attn2, i)
                        if a3 is not None:
                            attn3, __ = visualize_attn(I_test, a3, up_factor=4*opt.base_up_factor, nrow=8)
                            writer.add_image('test/attention_map_3', attn3, i)
    precision, recall, precision_mel, recall_mel = compute_mean_pecision_recall('test_results.csv')
    mAP, AUC, __ = compute_metrics('test_results.csv')
    print("\ntest result: accuracy %.2f%% \nmean precision %.2f%% mean recall %.2f%% \
            \nprecision for mel %.2f%% recall for mel %.2f%% \nmAP %.2f%% AUC %.4f\n" %
            (100*correct/total, 100*precision, 100*recall, 100*precision_mel, 100*recall_mel, 100*mAP, AUC))

if __name__ == "__main__":
    if opt.preprocess:
        preprocess_data(root_dir='../data_2017')
    main()
