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
from model_vgg_grid import AttnVGG
from model_res_1 import AttnResNet
from utilities_ import *
from data import preprocess_data, ISIC2018

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description="Attn-SKin-FocalLoss-test")

parser.add_argument("--preprocess", action='store_true', help="run preprocess_data")

parser.add_argument("--outf", type=str, default="log_test", help='path of log files')
parser.add_argument("--base_up_factor", type=int, default=8, help="number of epochs")

parser.add_argument("--model", type=str, default="VGGNet", help='VGGNet or ResNet')
parser.add_argument("--normalize_attn", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')
parser.add_argument("--no_attention", action='store_true', help='turn off attention')
parser.add_argument("--log_images", action='store_true', help='log images')

opt = parser.parse_args()

def main():
    # load data
    print('\nloading the dataset ...\n')
    im_size = 224
    transform_test = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        transforms.Normalize((0.7560,0.5222,0.5431), (0.0909, 0.1248, 0.1401))
    ])
    testset = ISIC2018(csv_file='test.csv', shuffle=False, rotate=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)
    print('done')

    # load network
    print('\nloading the model ...\n')
    if not opt.no_attention:
        print('\nturn on attention ...\n')
    else:
        print('\nturn off attention ...\n')

    if opt.model == 'VGGNet':
        print('\nbase model: VGGNet ...\n')
        net = AttnVGG(num_classes=7, attention=not opt.no_attention, normalize_attn=opt.normalize_attn)
    elif opt.model == 'ResNet':
        print('\nbase model: ResNet ...\n')
        net = AttnResNet(num_classes=7, attention=not opt.no_attention, normalize_attn=opt.normalize_attn)
    else:
        raise NotImplementedError("Invalid base model name!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0,1]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.load_state_dict(torch.load('net.pth'))
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
                images_test, labels_test = data
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
                idx = torch.eq(predict, labels_test).nonzero().cpu().numpy().squeeze(1)
                images_disp = images_test[idx,:,:,:]
                if opt.log_images:
                    I_test = utils.make_grid(images_disp, nrow=8, normalize=True, scale_each=True)
                    writer.add_image('test/image', I_test, i)
                    if not opt.no_attention:
                        if opt.normalize_attn:
                            vis_fun = visualize_attn_softmax
                        else:
                            vis_fun = visualize_attn_sigmoid
                        __, c1, c2, c3 = model.forward(images_disp)
                        if c1 is not None:
                            attn1, __ = vis_fun(I_test, c1, up_factor=opt.base_up_factor, nrow=8)
                            writer.add_image('test/attention_map_1', attn1, i)
                        if c2 is not None:
                            attn2, __ = vis_fun(I_test, c2, up_factor=2*opt.base_up_factor, nrow=8)
                            writer.add_image('test/attention_map_2', attn2, i)
                        if c3 is not None:
                            attn3, __ = vis_fun(I_test, c3, up_factor=4*opt.base_up_factor, nrow=8)
                            writer.add_image('test/attention_map_3', attn3, i)
    precision, recall = compute_mean_pecision_recall('test_results.csv')
    print("accuracy %.2f%% \nmean precision %.2f%% mean recall %.2f%%\n" % (100*correct/total, 100*np.mean(precision), 100*np.mean(recall)))

if __name__ == "__main__":
    if opt.preprocess:
        preprocess_data(root_dir='data_2018')
    main()
