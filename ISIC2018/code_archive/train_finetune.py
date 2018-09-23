import os
import csv
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from model2 import AttnVGG
from loss import FocalLoss
from data import preprocess_data, ISIC2018
from utilities import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description="Attn-Skin-Lesion")

parser.add_argument("--preprocess", type=bool, default=False, help="whether to run preprocess_data")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.01, help="initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')

parser.add_argument("--initialize", type=str, default="kaimingNormal", help='kaimingNormal or kaimingUniform or xavierNormal or xavierUniform')
parser.add_argument("--focal_loss", action='store_true', help='turn on focal loss (otherwise use cross entropy loss)')
parser.add_argument("--no_attention", action='store_true', help='turn off attention')
parser.add_argument("--over_sample", action='store_true', help='offline oversampling')

opt = parser.parse_args()

def main():
    # load data
    print('\nloading the dataset ...\n')
    if opt.over_sample:
        print('\ndata is offline oversampled ...\n')
        num_aug = 1
        train_file = 'train_oversample.csv'
    else:
        print('\nno offline oversampled ...\n')
        num_aug = 3
        train_file = 'train.csv'
    im_size = 256
    transform_train = transforms.Compose([
        transforms.Resize(300),
        transforms.RandomCrop(im_size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.7558,0.5230,0.5437), (0.1027, 0.1319, 0.1468))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        transforms.Normalize((0.7558,0.5230,0.5437), (0.1027, 0.1319, 0.1468))
    ])
    trainset = ISIC2018(csv_file=train_file, shuffle=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=6)
    testset = ISIC2018(csv_file='test.csv', shuffle=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=6)
    print('done')
    # load network
    print('\nloading the model ...\n')
    if not opt.no_attention:
        print('\nturn on attention ...\n')
    else:
        print('\nturn off attention ...\n')
    net = AttnVGG(num_classes=7, attention=not opt.no_attention, normalize_attn=True, init=opt.initialize)
    if opt.focal_loss:
        print('\nuse focal loss ...\n')
        criterion = FocalLoss(gama=2., size_average=True, weight=None)
    else:
        print('\nuse cross entropy loss ...\n')
        criterion = nn.CrossEntropyLoss()
    # move to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0,1]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    criterion.to(device)
    # initialize with pre-trained model
    model_dict = model.state_dict()
    pretrained_dict = torch.load('noattn_focal_over_100.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'module.classify' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    lr_lambda = lambda epoch : np.power(0.5, int(epoch/25))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print('done')

    # training
    print('\nstart training ...\n')
    step = 0
    running_avg_accuracy = 0
    writer = SummaryWriter(opt.outf)
    for epoch in range(opt.epochs):
        images_test_disp = []
        # adjust learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        print("\nepoch %d learning rate %f\n" % (epoch, current_lr))
        # run for one epoch
        for aug in range(num_aug):
            for i, data in enumerate(trainloader, 0):
                # warm up
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # make one-hot
                # if opt.focal_loss:
                #     labels_one_hot = torch.zeros(labels.size(0),7).to(device).scatter_(1, labels.view(-1,1), 1.)
                if i == 0: # archive images in order to save to logs
                    images_test_disp.append(inputs[0:16,:,:,:])
                # forward
                pred, __, __, __ = model.forward(inputs)
                # backward
                loss = criterion(pred, labels)
                # loss.backward()
                loss.backward()
                optimizer.step()
                # display results
                if i % 10 == 0:
                    model.eval()
                    pred, __, __, __ = model.forward(inputs)
                    predict = torch.argmax(pred, 1)
                    total = labels.size(0)
                    correct = torch.eq(predict, labels).sum().double().item()
                    accuracy = correct / total
                    running_avg_accuracy = 0.99*running_avg_accuracy + 0.01*accuracy
                    writer.add_scalar('train/loss', loss.item(), step)
                    writer.add_scalar('train/accuracy', accuracy, step)
                    writer.add_scalar('train/running_avg_accuracy', running_avg_accuracy, step)
                    print("[epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%"
                        % (epoch, aug, num_aug-1, i, len(trainloader)-1, loss.item(), (100*accuracy), (100*running_avg_accuracy)))
                step += 1
        # the end of each epoch: test & log
        model.eval()
        print('\none epoch done, saving checkpoints ...\n')
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
        if epoch == opt.epochs / 2:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net%d.pth' % epoch))
        total = 0
        correct = 0
        with torch.no_grad():
            with open('test_results.csv', 'wt', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                for i, data in enumerate(testloader, 0):
                    images_test, labels_test = data
                    images_test, labels_test = images_test.to(device), labels_test.to(device)
                    if i == 0: # archive images in order to save to logs
                        images_test_disp.append(images_test[0:16,:,:,:])
                    pred_test, __, __, __ = model.forward(images_test)
                    predict = torch.argmax(pred_test, 1)
                    total += labels_test.size(0)
                    correct += torch.eq(predict, labels_test).sum().double().item()
                    # record test predicted responses
                    responses = F.softmax(pred_test, dim=1).squeeze().cpu().numpy()
                    responses = [responses[i] for i in range(responses.shape[0])]
                    csv_writer.writerows(responses)
            # log scalars
            precision = avg_precision('test_results.csv')
            recall = avg_recall('test_results.csv')
            writer.add_scalar('test/accuracy', correct/total, epoch)
            writer.add_scalar('test/avg_precision', precision, epoch)
            writer.add_scalar('test/avg_recall', recall, epoch)
            print("\n[epoch %d] test result: accuracy %.2f%% \navg_precision %.2f%% avg_recall %.2f%%\n" %
                (epoch, 100*correct/total, 100*precision, 100*recall))
            # log images
            if not opt.no_attention:
                # training data
                __, c1, c2, c3 = model.forward(images_test_disp[0])
                I = utils.make_grid(images_test_disp[0], nrow=4, normalize=True, scale_each=True)
                attn1 = visualize_attn_softmax(I, c1, up_factor=8, nrow=4)
                attn2 = visualize_attn_softmax(I, c2, up_factor=16, nrow=4)
                writer.add_image('train/image', I, epoch)
                writer.add_image('train/attention_map_1', attn1, epoch)
                writer.add_image('train/attention_map_2', attn2, epoch)
                if c3 is not None:
                    attn3 = visualize_attn_softmax(I, c3, up_factor=32, nrow=4)
                    writer.add_image('train/attention_map_3', attn3, epoch)
                # test data
                __, c1, c2, c3 = model.forward(images_test_disp[1])
                I = utils.make_grid(images_test_disp[1], nrow=4, normalize=True, scale_each=True)
                attn1 = visualize_attn_softmax(I, c1, up_factor=8, nrow=4)
                attn2 = visualize_attn_softmax(I, c2, up_factor=16, nrow=4)
                writer.add_image('test/image', I, epoch)
                writer.add_image('test/attention_map_1', attn1, epoch)
                writer.add_image('test/attention_map_2', attn2, epoch)
                if c3 is not None:
                    attn3 = visualize_attn_softmax(I, c3, up_factor=32, nrow=4)
                    writer.add_image('test/attention_map_3', attn3, epoch)

if __name__ == "__main__":
    if opt.preprocess:
        preprocess_data(root_dir='data')
    main()
