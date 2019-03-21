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

parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.01, help="initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--base_up_factor", type=int, default=8, help="upsample ratio for attention visualization")

parser.add_argument("--normalize_attn", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')
parser.add_argument("--focal_loss", action='store_true', help='turn on focal loss (otherwise use cross entropy loss)')
parser.add_argument("--no_attention", action='store_true', help='turn off attention')
parser.add_argument("--over_sample", action='store_true', help='offline oversampling')
parser.add_argument("--log_images", action='store_true', help='visualze images in Tensoeboard')

opt = parser.parse_args()

def _worker_init_fn_():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32-1
    random.seed(torch_seed)
    np.random.seed(np_seed)

def main():
    # load data
    print('\nloading the dataset ...')
    assert opt.dataset == "ISIC2016" or opt.dataset == "ISIC2017"
    if opt.dataset == "ISIC2016":
        num_aug = 5
        normalize = Normalize((0.7012, 0.5517, 0.4875), (0.0942, 0.1331, 0.1521))
    elif opt.dataset == "ISIC2017":
        num_aug = 2
        normalize = Normalize((0.6820, 0.5312, 0.4736), (0.0840, 0.1140, 0.1282))
    if opt.over_sample:
        print('data is offline oversampled ...')
        train_file = 'train_oversample.csv'
    else:
        print('no offline oversampling ...')
        train_file = 'train.csv'
    transform_train = torch_transforms.Compose([
         RatioCenterCrop(0.8),
         Resize((256,256)),
         RandomCrop((224,224)),
         RandomRotate(),
         RandomHorizontalFlip(),
         RandomVerticalFlip(),
         ToTensor(),
         normalize
    ])
    transform_val = torch_transforms.Compose([
         RatioCenterCrop(0.8),
         Resize((256,256)),
         CenterCrop((224,224)),
         ToTensor(),
         normalize
    ])
    trainset = ISIC(csv_file=train_file, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
        num_workers=8, worker_init_fn=_worker_init_fn_(), drop_last=True)
    valset = ISIC(csv_file='val.csv', transform=transform_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=8)
    print('done\n')

    # load models
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

    if opt.focal_loss:
        print('use focal loss ...')
        criterion = FocalLoss(gama=2., size_average=True, weight=None)
    else:
        print('use cross entropy loss ...')
        criterion = nn.CrossEntropyLoss()
    print('done\n')

    # move to GPU
    print('\nmoving models to GPU ...')
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    criterion.to(device)
    print('done\n')

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    lr_lambda = lambda epoch : np.power(0.1, epoch//10)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # training
    print('\nstart training ...\n')
    step = 0
    EMA_accuracy = 0
    AUC_val = 0
    writer = SummaryWriter(opt.outf)
    if opt.log_images:
        data_iter = iter(valloader)
        fixed_batch = next(data_iter)
        fixed_batch = fixed_batch['image'][0:16,:,:,:].to(device)
    for epoch in range(opt.epochs):
        torch.cuda.empty_cache()
        # adjust learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        print("\nepoch %d learning rate %f\n" % (epoch+1, current_lr))
        # run for one epoch
        for aug in range(num_aug):
            for i, data in enumerate(trainloader, 0):
                # warm up
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                inputs, labels = data['image'], data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                # forward
                pred, __, __ = model(inputs)
                # backward
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                # display results
                if i % 10 == 0:
                    model.eval()
                    pred, __, __ = model(inputs)
                    predict = torch.argmax(pred, 1)
                    total = labels.size(0)
                    correct = torch.eq(predict, labels).sum().double().item()
                    accuracy = correct / total
                    EMA_accuracy = 0.9*EMA_accuracy + 0.1*accuracy
                    writer.add_scalar('train/loss_c', loss.item(), step)
                    writer.add_scalar('train/accuracy', accuracy, step)
                    writer.add_scalar('train/EMA_accuracy', EMA_accuracy, step)
                    print("[epoch %d][aug %d/%d][iter %d/%d] loss %.4f accuracy %.2f%% EMA_accuracy %.2f%%"
                        % (epoch+1, aug+1, num_aug, i+1, len(trainloader), loss.item(), (100*accuracy), (100*EMA_accuracy)))
                step += 1
        # the end of each epoch - validation results
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            with open('val_results.csv', 'wt', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                for i, data in enumerate(valloader, 0):
                    images_val, labels_val = data['image'], data['label']
                    images_val, labels_val = images_val.to(device), labels_val.to(device)
                    pred_val, __, __ = model(images_val)
                    predict = torch.argmax(pred_val, 1)
                    total += labels_val.size(0)
                    correct += torch.eq(predict, labels_val).sum().double().item()
                    # record prediction
                    responses = F.softmax(pred_val, dim=1).squeeze().cpu().numpy()
                    responses = [responses[i] for i in range(responses.shape[0])]
                    csv_writer.writerows(responses)
            AP, AUC, precision_mean, precision_mel, recall_mean, recall_mel = compute_metrics('val_results.csv', 'val.csv')
            # save checkpoints
            print('\nsaving checkpoints ...\n')
            checkpoint = {
                'state_dict': model.module.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(opt.outf, 'checkpoint_latest.pth'))
            if AUC > AUC_val: # save optimal validation model
                torch.save(checkpoint, os.path.join(opt.outf,'checkpoint.pth'))
                AUC_val = AUC
            # log scalars
            writer.add_scalar('val/accuracy', correct/total, epoch)
            writer.add_scalar('val/mean_precision', precision_mean, epoch)
            writer.add_scalar('val/mean_recall', recall_mean, epoch)
            writer.add_scalar('val/precision_mel', precision_mel, epoch)
            writer.add_scalar('val/recall_mel', recall_mel, epoch)
            writer.add_scalar('val/AP', AP, epoch)
            writer.add_scalar('val/AUC', AUC, epoch)
            print("\n[epoch %d] val result: accuracy %.2f%%" % (epoch+1, 100*correct/total))
            print("\nmean precision %.2f%% mean recall %.2f%% \nprecision for mel %.2f%% recall for mel %.2f%%" %
                    (100*precision_mean, 100*recall_mean, 100*precision_mel, 100*recall_mel))
            print("\nAP %.4f AUC %.4f optimal AUC: %.4f\n" % (AP, AUC, AUC_val))
            # log images
            if opt.log_images:
                print('\nlog images ...\n')
                I_train = utils.make_grid(inputs[0:16,:,:,:], nrow=4, normalize=True, scale_each=True)
                writer.add_image('train/image', I_train, epoch)
                if epoch == 0:
                    I_val = utils.make_grid(fixed_batch, nrow=4, normalize=True, scale_each=True)
                    writer.add_image('val/image', I_val, epoch)
            if opt.log_images and (not opt.no_attention):
                print('\nlog attention maps ...\n')
                # training data
                __, a1, a2 = model(inputs[0:16,:,:,:])
                if a1 is not None:
                    attn1 = visualize_attn(I_train, a1, up_factor=opt.base_up_factor, nrow=4)
                    writer.add_image('train/attention_map_1', attn1, epoch)
                if a2 is not None:
                    attn2 = visualize_attn(I_train, a2, up_factor=2*opt.base_up_factor, nrow=4)
                    writer.add_image('train/attention_map_2', attn2, epoch)
                # val data
                __, a1, a2 = model(fixed_batch)
                if a1 is not None:
                    attn1 = visualize_attn(I_val, a1, up_factor=opt.base_up_factor, nrow=4)
                    writer.add_image('val/attention_map_1', attn1, epoch)
                if a2 is not None:
                    attn2 = visualize_attn(I_val, a2, up_factor=2*opt.base_up_factor, nrow=4)
                    writer.add_image('val/attention_map_2', attn2, epoch)

if __name__ == "__main__":
    if opt.preprocess:
        if opt.dataset == "ISIC2016":
            preprocess_data_2016(root_dir='../data_2016')
        elif opt.dataset == "ISIC2017":
            preprocess_data_2017(root_dir='../data_2017', seg_dir='Train_Lesion')
    main()
