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
from model5 import AttnVGG
from model6 import AttnResNet
from utilities import *
from data import preprocess_data, ISIC2017

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description="Attn-SKin-FocalLoss-test")

parser.add_argument("--preprocess", type=bool, default=False, help="whether to run preprocess_data")
parser.add_argument("--outf", type=str, default="logs_test", help='path of log files')
parser.add_argument("--base_up_factor", type=int, default=8, help="number of epochs")

parser.add_argument("--model", type=str, default="VGGNet", help='VGGNet or ResNet')
parser.add_argument("--no_attention", action='store_true', help='turn off attention')
parser.add_argument("--log_images", action='store_true', help='log images')

opt = parser.parse_args()

def main():
    # prepare for CAM
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.cpu().numpy())

    # load data
    print('\nloading the dataset ...\n')
    im_size = 256
    transform_test = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        transforms.Normalize((0.7540,0.5188,0.5392), (0.0917, 0.1255, 0.1408))
    ])
    testset = ISIC2017(csv_file='test.csv', shuffle=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=6)
    print('done')

    # load network
    print('\nloading the model ...\n')
    if not opt.no_attention:
        print('\nturn on attention ...\n')
    else:
        print('\nturn off attention ...\n')

    if opt.model == 'VGGNet':
        print('\nbase model: VGGNet ...\n')
        net = AttnVGG(num_classes=2, attention=not opt.no_attention, normalize_attn=True)
        net._modules.get('feature')._modules.get('0').register_forward_hook(hook_feature)
    elif opt.model == 'ResNet':
        print('\nbase model: ResNet ...\n')
        net = AttnResNet(num_classes=2, attention=not opt.no_attention, normalize_attn=True)
        net._modules.get('feature')._modules.get('1').register_forward_hook(hook_feature)
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
                features_blobs.clear()
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
                if opt.log_images:
                    I_test = utils.make_grid(images_test, nrow=8, normalize=True, scale_each=True)
                    writer.add_image('test/image', I_test, i)
                    # class activation map
                    params = list(model.parameters())
                    cam = returnCAM(I_test, feature_conv=features_blobs, weight_softmax=params[-2].cpu().numpy(), class_idx=1, im_size=im_size, nrow=8)
                    writer.add_image('test/CAM', cam, i)
                    if not opt.no_attention:
                        __, c1, c2, c3 = model.forward(images_test)
                        if c1 is not None:
                            attn1 = visualize_attn_softmax(I_test, c1, up_factor=opt.base_up_factor, nrow=8)
                            writer.add_image('test/attention_map_1', attn1, i)
                        if c2 is not None:
                            attn2 = visualize_attn_softmax(I_test, c2, up_factor=2*opt.base_up_factor, nrow=8)
                            writer.add_image('test/attention_map_2', attn2, i)
                        if c3 is not None:
                            attn3 = visualize_attn_softmax(I_test, c3, up_factor=4*opt.base_up_factor, nrow=8)
                            writer.add_image('test/attention_map_3', attn3, i)
    mAP, AUC, __ = compute_metrics('test_results.csv')
    precision, recall = compute_mean_pecision_recall('test_results.csv')
    print("\ntest result: accuracy %.2f%% \nmean precision %.2f%% mean recall %.2f%% \nmAP %.2f%% AUC %.4f\n" %
        (100*correct/total, 100*precision, 100*recall, 100*mAP, AUC))

if __name__ == "__main__":
    if opt.preprocess:
        preprocess_data(root_dir='data_2017')
    main()
