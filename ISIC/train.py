import os
import csv
import random
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
from networks import AttnVGG
from loss import FocalLoss
from data_2017 import preprocess_data, ISIC
from utilities import *

'''
switch between ISIC 2016 and 2017
modify the following contents:
1. import
2. num_aug
3. root_dir of preprocess_data
4. mean and std of transforms.Normalize
'''

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0,1]

parser = argparse.ArgumentParser(description="Attn-Skin-train")

parser.add_argument("--preprocess", action='store_true', help="run preprocess_data")

parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.01, help="initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--base_up_factor", type=int, default=8, help="number of epochs")

parser.add_argument("--normalize_attn", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')
parser.add_argument("--focal_loss", action='store_true', help='turn on focal loss (otherwise use cross entropy loss)')
parser.add_argument("--no_attention", action='store_true', help='turn off attention')
parser.add_argument("--over_sample", action='store_true', help='offline oversampling')
parser.add_argument("--log_images", action='store_true', help='offline oversampling')

opt = parser.parse_args()

def __worker_init_fn__():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32-1
    random.seed(torch_seed)
    np.random.seed(np_seed)

def main():
    # load data
    print('\nloading the dataset ...\n')
    if opt.over_sample:
        print('\ndata is offline oversampled ...\n')
        num_aug = 5
        train_file = 'train_oversample.csv'
    else:
        print('\nno offline oversampling ...\n')
        num_aug = 8
        train_file = 'train.csv'
    im_size = 224
    transform_test = transforms.Compose([
        transforms.CenterCrop(im_size),
        transforms.ToTensor()
    ])
    trainset = ISIC(csv_file=train_file, shuffle=True, resize=(256,256), randcrop=(im_size,im_size), rotate=True, flip=True,
        transform=transforms.ToTensor(), transform_seg=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8, worker_init_fn=__worker_init_fn__())
    testset = ISIC(csv_file='test.csv', shuffle=False, resize=(256,256), randcrop=None, rotate=False, flip=False,
        transform=transform_test, transform_seg=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

    # ISIC 2017 (full crop): mean & std of the dataset
    # Mean = torch.from_numpy(np.array([0.7061, 0.5725, 0.5200]).astype(np.float32))
    # Std  = torch.from_numpy(np.array([0.0810, 0.1101, 0.1256]).astype(np.float32))
    # ISIC 2017 (0.8 crop): mean & std of the dataset
    Mean = torch.from_numpy(np.array([0.6900, 0.5442, 0.4865]).astype(np.float32))
    Std  = torch.from_numpy(np.array([0.0810, 0.1118, 0.1265]).astype(np.float32))
    # ISIC 2016: mean & std of the dataset
    # Mean = torch.from_numpy(np.array([0.7105, 0.5646, 0.4978]).astype(np.float32))
    # Std  = torch.from_numpy(np.array([0.0910, 0.1310, 0.1510]).astype(np.float32))
    '''
    Mean = torch.zeros(3)
    Std = torch.zeros(3)
    for data in trainloader:
        I, __, L = data
        N, C, __, __ = I.size()
        Mean += I.view(N,C,-1).mean(2).sum(0)
        Std += I.view(N,C,-1).std(2).sum(0)
    Mean /= len(trainset)
    Std  /= len(trainset)
    print('mean: '), print(Mean.numpy())
    print('std: '), print(Std.numpy())
    return
    '''
    print('\ndone\n')

    # load models
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

    if opt.focal_loss:
        print('\nuse focal loss ...\n')
        criterion = FocalLoss(gama=2., size_average=True, weight=None)
    else:
        print('\nuse cross entropy loss ...\n')
        criterion = nn.CrossEntropyLoss()
    print('\ndone\n')

    # move to GPU
    print('\nmoving models to GPU ...\n')
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    criterion.to(device)
    print('\ndone\n')

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    lr_lambda = lambda epoch : np.power(0.5, epoch//10)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # training
    print('\nstart training ...\n')
    step = 0
    EMA_accuracy = 0
    writer = SummaryWriter(opt.outf)
    for epoch in range(opt.epochs):
        images_disp = []
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
                inputs, __, labels = data
                inputs = (inputs - Mean.view(1,3,1,1)) / Std.view(1,3,1,1)
                inputs, labels = inputs.to(device), labels.to(device)
                if (aug == 0) and (i == 0): # archive images in order to save to logs
                    images_disp.append(inputs[0:16,:,:,:])
                # forward
                pred, __, __, __ = model.forward(inputs)
                # backward
                loss = criterion(pred, labels)
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
                    EMA_accuracy = 0.98*EMA_accuracy + 0.02*accuracy
                    writer.add_scalar('train/loss_c', loss.item(), step)
                    writer.add_scalar('train/accuracy', accuracy, step)
                    writer.add_scalar('train/EMA_accuracy', EMA_accuracy, step)
                    print("[epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% EMA_accuracy %.2f%%"
                        % (epoch, aug, num_aug-1, i, len(trainloader)-1, loss.item(), (100*accuracy), (100*EMA_accuracy)))
                step += 1
        # the end of each epoch
        model.eval()
        # save checkpoints
        print('\none epoch done, saving checkpoints ...\n')
        checkpoint = {
            'state_dict': model.module.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(opt.outf,'checkpoint.pth'))
        if epoch == opt.epochs / 2:
            torch.save(checkpoint, os.path.join(opt.outf, 'checkpoint_%d.pth' % epoch))
        # log test results
        total = 0
        correct = 0
        with torch.no_grad():
            with open('test_results.csv', 'wt', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                for i, data in enumerate(testloader, 0):
                    images_test, __, labels_test = data
                    images_test = (images_test - Mean.view(1,3,1,1)) / Std.view(1,3,1,1)
                    images_test, labels_test = images_test.to(device), labels_test.to(device)
                    if i == 0: # archive images in order to save to logs
                        images_disp.append(images_test[0:16,:,:,:])
                    pred_test, __, __, __ = model.forward(images_test)
                    predict = torch.argmax(pred_test, 1)
                    total += labels_test.size(0)
                    correct += torch.eq(predict, labels_test).sum().double().item()
                    # record test predicted responses
                    responses = F.softmax(pred_test, dim=1).squeeze().cpu().numpy()
                    responses = [responses[i] for i in range(responses.shape[0])]
                    csv_writer.writerows(responses)
            # log scalars
            precision, recall, precision_mel, recall_mel = compute_mean_pecision_recall('test_results.csv')
            mAP, AUC, ROC = compute_metrics('test_results.csv')
            writer.add_scalar('test/accuracy', correct/total, epoch)
            writer.add_scalar('test/mean_precision', precision, epoch)
            writer.add_scalar('test/mean_recall', recall, epoch)
            writer.add_scalar('test/precision_mel', precision_mel, epoch)
            writer.add_scalar('test/recall_mel', recall_mel, epoch)
            writer.add_scalar('test/mAP', mAP, epoch)
            writer.add_scalar('test/AUC', AUC, epoch)
            writer.add_image('curve/ROC', ROC, epoch)
            print("\n[epoch %d] test result: accuracy %.2f%% \nmean precision %.2f%% mean recall %.2f%% \
                    \nprecision for mel %.2f%% recall for mel %.2f%% \nmAP %.2f%% AUC %.4f\n" %
                    (epoch, 100*correct/total, 100*precision, 100*recall, 100*precision_mel, 100*recall_mel, 100*mAP, AUC))
            # log images
            if opt.log_images:
                print('\nlog images ...\n')
                I_train = utils.make_grid(images_disp[0], nrow=4, normalize=True, scale_each=True)
                writer.add_image('train/image', I_train, epoch)
                if epoch == 0:
                    I_test = utils.make_grid(images_disp[1], nrow=4, normalize=True, scale_each=True)
                    writer.add_image('test/image', I_test, epoch)
            if opt.log_images and (not opt.no_attention):
                print('\nlog attention maps ...\n')
                # training data
                __, a1, a2, a3 = model.forward(images_disp[0])
                if a1 is not None:
                    attn1, stat = visualize_attn(I_train, a1, up_factor=opt.base_up_factor, nrow=4)
                    writer.add_image('train/attention_map_1', attn1, epoch)
                    writer.add_scalar('train_a1/max', stat[0], epoch)
                    writer.add_scalar('train_a1/min', stat[1], epoch)
                    writer.add_scalar('train_a1/mean', stat[2], epoch)
                if a2 is not None:
                    attn2, stat = visualize_attn(I_train, a2, up_factor=2*opt.base_up_factor, nrow=4)
                    writer.add_image('train/attention_map_2', attn2, epoch)
                    writer.add_scalar('train_a2/max', stat[0], epoch)
                    writer.add_scalar('train_a2/min', stat[1], epoch)
                    writer.add_scalar('train_a2/mean', stat[2], epoch)
                if a3 is not None:
                    attn3, stat = visualize_attn(I_train, a3, up_factor=4*opt.base_up_factor, nrow=4)
                    writer.add_image('train/attention_map_3', attn3, epoch)
                    writer.add_scalar('train_a3/max', stat[0], epoch)
                    writer.add_scalar('train_a3/min', stat[1], epoch)
                    writer.add_scalar('train_a3/mean', stat[2], epoch)
                # test data
                __, a1, a2, a3 = model.forward(images_disp[1])
                if a1 is not None:
                    attn1, __ = visualize_attn(I_test, a1, up_factor=opt.base_up_factor, nrow=4)
                    writer.add_image('test/attention_map_1', attn1, epoch)
                if a2 is not None:
                    attn2, __ = visualize_attn(I_test, a2, up_factor=2*opt.base_up_factor, nrow=4)
                    writer.add_image('test/attention_map_2', attn2, epoch)
                if a3 is not None:
                    attn3, __ = visualize_attn(I_test, a3, up_factor=4*opt.base_up_factor, nrow=4)
                    writer.add_image('test/attention_map_3', attn3, epoch)

if __name__ == "__main__":
    if opt.preprocess:
        preprocess_data(root_dir='../data_2017')
    main()
