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
from model2 import AttnVGG
from utilities import *
from data import preprocess_data, ISIC2018

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description="Attn-SKin-FocalLoss-test")
parser.add_argument("--preprocess", type=bool, default=False, help="whether to run preprocess_data")
parser.add_argument("--outf", type=str, default="log_test", help='path of log files')
parser.add_argument("--no_attention", action='store_true', help='turn off attention')

opt = parser.parse_args()

def main():
    # load data
    print('\nloading the dataset ...\n')
    im_size = 256
    transform_test = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        transforms.Normalize((0.7558,0.5230,0.5437), (0.1027, 0.1319, 0.1468))
    ])
    testset = ISIC2018(csv_file='test.csv', shuffle=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=6)
    print('done')

    # load network
    print('\nloading the model ...\n')
    if not opt.no_attention:
        print('\nturn on attention ...\n')
    else:
        print('\nturn off attention ...\n')
    net = AttnVGG(num_classes=7, attention=not opt.no_attention, normalize_attn=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0,1]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.load_state_dict(torch.load('net.pth'))
    model.eval()
    print('done')

    # testing
    print('\nstart testing ...\n')
    writer = SummaryWriter(opt.outf)
    with torch.no_grad():
        with open('test_results_.csv', 'wt', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            for i, data in enumerate(testloader, 0):
                images_test, labels_test = data
                images_test, labels_test = images_test.to(device), labels_test.to(device)
                pred_test, __, __, __ = model.forward(images_test)
                # record test predicted responses
                responses = F.softmax(pred_test, dim=1).squeeze().cpu().numpy()
                responses = [responses[i] for i in range(responses.shape[0])]
                csv_writer.writerows(responses)
                # log images
                if not opt.no_attention:
                    __, c1, c2, c3 = model.forward(images_test[0:16,:,:,:])
                    I = utils.make_grid(images_test[0:16,:,:,:], nrow=4, normalize=True, scale_each=True)
                    attn1 = visualize_attn_softmax(I, c1, up_factor=8, nrow=4)
                    attn2 = visualize_attn_softmax(I, c2, up_factor=16, nrow=4)
                    writer.add_image('test/image', I, epoch)
                    writer.add_image('test/attention_map_1', attn1, epoch)
                    writer.add_image('test/attention_map_2', attn2, epoch)
                    if c3 is not None:
                        attn3 = visualize_attn_softmax(I, c3, up_factor=32, nrow=4)
                        writer.add_image('test/attention_map_3', attn3, epoch)
    precision, recall = compute_mean_pecision_recall('test_results.csv')
    print("\navg_precision %.2f%% avg_recall %.2f%%\n" % (100*precision, 100*recall))

if __name__ == "__main__":
    if opt.preprocess:
        preprocess_data(root_dir='data')
    main()
