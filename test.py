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
import torchvision.transforms as torch_transforms
from networks import AttnVGG, VGG
from loss import FocalLoss
from data import preprocess_data_2016, preprocess_data_2017, ISIC
from utilities import *
from transforms import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

parser = argparse.ArgumentParser()

parser.add_argument("--preprocess", action='store_true', help="run preprocess_data")
parser.add_argument("--dataset", type=str, default="ISIC2017", help='ISIC2017 / ISIC2016')

parser.add_argument("--outf", type=str, default="logs_test", help='path of log files')
parser.add_argument("--base_up_factor", type=int, default=8, help="number of epochs")

parser.add_argument("--normalize_attn", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')
parser.add_argument("--no_attention", action='store_true', help='turn off attention')
parser.add_argument("--log_images", action='store_true', help='visualze images in Tensoeboard')

opt = parser.parse_args()

def main():
    # load data
    print('\nloading the dataset ...')
    assert opt.dataset == "ISIC2016" or opt.dataset == "ISIC2017"
    if opt.dataset == "ISIC2016":
        normalize = Normalize((0.7012, 0.5517, 0.4875), (0.0942, 0.1331, 0.1521))
    elif opt.dataset == "ISIC2017":
        normalize = Normalize((0.6820, 0.5312, 0.4736), (0.0840, 0.1140, 0.1282))
    transform_test = torch_transforms.Compose([
         RatioCenterCrop(0.8),
         Resize((256,256)),
         CenterCrop((224,224)),
         ToTensor(),
         normalize
    ])
    testset = ISIC(csv_file='test.csv', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)
    print('done\n')

    # load network
    print('\nloading the model ...')
    if not opt.no_attention:
        print('turn on attention ...')
        if opt.normalize_attn:
            print('use softmax for attention map ...')
        else:
            print('use sigmoid for attention map ...')
    else:
        print('turn off attention ...')

    net = AttnVGG(num_classes=2, attention=not opt.no_attention, normalize_attn=opt.normalize_attn)
    # net = VGG(num_classes=2, gap=False)
    checkpoint = torch.load('checkpoint.pth')
    net.load_state_dict(checkpoint['state_dict'])
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.eval()
    print('done\n')

    # testing
    print('\nstart testing ...\n')
    writer = SummaryWriter(opt.outf)
    total = 0
    correct = 0
    with torch.no_grad():
        with open('test_results.csv', 'wt', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for i, data in enumerate(testloader, 0):
                images_test, labels_test = data['image'], data['label']
                images_test, labels_test = images_test.to(device), labels_test.to(device)
                pred_test, __, __ = model.forward(images_test)
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
                        __, a1, a2 = model.forward(images_test)
                        if a1 is not None:
                            attn1 = visualize_attn(I_test, a1, up_factor=opt.base_up_factor, nrow=8)
                            writer.add_image('test/attention_map_1', attn1, i)
                        if a2 is not None:
                            attn2 = visualize_attn(I_test, a2, up_factor=2*opt.base_up_factor, nrow=8)
                            writer.add_image('test/attention_map_2', attn2, i)
    AP, AUC, precision_mean, precision_mel, recall_mean, recall_mel = compute_metrics('test_results.csv', 'test.csv')
    print("\ntest result: accuracy %.2f%%" % (100*correct/total))
    print("\nmean precision %.2f%% mean recall %.2f%% \nprecision for mel %.2f%% recall for mel %.2f%%" %
            (100*precision_mean, 100*recall_mean, 100*precision_mel, 100*recall_mel))
    print("\nAP %.4f AUC %.4f\n" % (AP, AUC))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.dataset == "ISIC2016":
            preprocess_data_2016(root_dir='../data_2016')
        elif opt.dataset == "ISIC2017":
            preprocess_data_2017(root_dir='../data_2017', seg_dir='Train_Lesion')
    main()
