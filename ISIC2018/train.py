import os
import csv
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from networks import AttnVGG, VGG
from loss import FocalLoss
from data import preprocess_data, ISIC2018
from utilities import *
from transforms import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

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
parser.add_argument("--log_images", action='store_true', help='log images')

opt = parser.parse_args()

def _worker_init_fn_():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32-1
    random.seed(torch_seed)
    np.random.seed(np_seed)

def main():
    # load data
    print('\nloading the dataset ...\n')
    if opt.over_sample:
        print('\ndata is offline oversampled ...\n')
        num_aug = 1
        train_file = 'train_oversample.csv'
    else:
        print('\nno offline oversampled ...\n')
        num_aug = 5
        train_file = 'train.csv'
    transform_train = transforms.Compose([
        RatioCenterCrop(1.0),
        Resize((256,256)),
        RandomCrop((224,224)),
        RandomRotate(),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.7560,0.5222,0.5431), (0.0909, 0.1248, 0.1400))
    ])
    transform_val = transforms.Compose([
        RatioCenterCrop(1.0),
        Resize((256,256)),
        CenterCrop((224,224)),
        ToTensor(),
        Normalize((0.7560,0.5222,0.5431), (0.0909, 0.1248, 0.1400))
    ])
    trainset = ISIC2018(csv_file=train_file, shuffle=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8, worker_init_fn=_worker_init_fn_())
    valset = ISIC2018(csv_file='val.csv', shuffle=False, transform=transform_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=8)
    # mean & std of the datase
    '''
    Mean = torch.zeros(3)
    Std  = torch.zeros(3)
    for data in trainloader:
        I = data['image']
        N, C, __, __ = I.size()
        Mean += I.view(N,C,-1).mean(2).sum(0)
        Std += I.view(N,C,-1).std(2).sum(0)
    Mean /= len(trainset)
    Std /= len(trainset)
    print('mean: '), print(Mean.numpy())
    print('std: '), print(Std.numpy())
    return
    '''
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

    net = AttnVGG(num_classes=7, attention=not opt.no_attention, normalize_attn=opt.normalize_attn)
    # net = VGG(num_classes=7, gap=False)

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
        torch.cuda.empty_cache()
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
                inputs, labels = data['image'], data['label']
                inputs, labels = inputs.to(device), labels.to(device)
                if (aug == 0) and (i == 0): # archive images in order to save to logs
                    images_disp.append(inputs[0:16,:,:,:])
                # forward
                pred, __, __ = model.forward(inputs)
                # backward
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                # display results
                if i % 10 == 0:
                    model.eval()
                    pred, __, __ = model.forward(inputs)
                    predict = torch.argmax(pred, 1)
                    total = labels.size(0)
                    correct = torch.eq(predict, labels).sum().double().item()
                    accuracy = correct / total
                    EMA_accuracy = 0.98*EMA_accuracy + 0.02*accuracy
                    writer.add_scalar('train/loss', loss.item(), step)
                    writer.add_scalar('train/accuracy', accuracy, step)
                    writer.add_scalar('train/EMA_accuracy', EMA_accuracy, step)
                    print("[epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%"
                        % (epoch, aug, num_aug-1, i, len(trainloader)-1, loss.item(), (100*accuracy), (100*EMA_accuracy)))
                step += 1
        # the end of each epoch
        model.eval()
        print('\none epoch done, saving checkpoints ...\n')
        checkpoint = {
            'state_dict': model.module.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(opt.outf,'checkpoint.pth'))
        if epoch % 10 == 9:
            torch.save(checkpoint, os.path.join(opt.outf, 'checkpoint_%d.pth' % epoch))
        total = 0
        correct = 0
        with torch.no_grad():
            with open('val_results.csv', 'wt', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                for i, data in enumerate(valloader, 0):
                    images_val, labels_val = data['image'], data['label']
                    images_val, labels_val = images_val.to(device), labels_val.to(device)
                    if i == 0: # archive images in order to save to logs
                        images_disp.append(images_val[0:16,:,:,:])
                    pred_val, __, __ = model.forward(images_val)
                    predict = torch.argmax(pred_val, 1)
                    total += labels_val.size(0)
                    correct += torch.eq(predict, labels_val).sum().double().item()
                    # record prediction
                    responses = F.softmax(pred_val, dim=1).squeeze().cpu().numpy()
                    responses = [responses[i] for i in range(responses.shape[0])]
                    csv_writer.writerows(responses)
            # log scalars
            precision, recall = compute_mean_pecision_recall('val_results.csv', 'val.csv')
            writer.add_scalar('val/accuracy', correct/total, epoch)
            writer.add_scalar('val/mean_precision', np.mean(precision), epoch)
            writer.add_scalar('val/mean_recall', np.mean(recall), epoch)
            ####
            print(precision)
            print(recall)
            writer.add_scalar('precision/MEL',   precision[0], epoch)
            writer.add_scalar('precision/NV',    precision[1], epoch)
            writer.add_scalar('precision/BCC',   precision[2], epoch)
            writer.add_scalar('precision/AKIEC', precision[3], epoch)
            writer.add_scalar('precision/BKL',   precision[4], epoch)
            writer.add_scalar('precision/DF',    precision[5], epoch)
            writer.add_scalar('precision/VASC',  precision[6], epoch)
            writer.add_scalar('recall/MEL',   recall[0], epoch)
            writer.add_scalar('recall/NV',    recall[1], epoch)
            writer.add_scalar('recall/BCC',   recall[2], epoch)
            writer.add_scalar('recall/AKIEC', recall[3], epoch)
            writer.add_scalar('recall/BKL',   recall[4], epoch)
            writer.add_scalar('recall/DF',    recall[5], epoch)
            writer.add_scalar('recall/VASC',  recall[6], epoch)
            ####
            print("\n[epoch %d] val result: accuracy %.2f%% \nmean precision %.2f%% mean recall %.2f%%\n" %
                (epoch, 100*correct/total, 100*np.mean(precision), 100*np.mean(recall)))
            # log images
            if opt.log_images:
                print('\nlog images ...\n')
                I_train = utils.make_grid(images_disp[0], nrow=4, normalize=True, scale_each=True)
                writer.add_image('train/image', I_train, epoch)
                if epoch == 0:
                    I_val = utils.make_grid(images_disp[1], nrow=4, normalize=True, scale_each=True)
                    writer.add_image('val/image', I_val, epoch)
            if opt.log_images and (not opt.no_attention):
                print('\nlog attention maps ...\n')
                # training data
                __, a1, a2 = model.forward(images_disp[0])
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
                # val data
                __, a1, a2 = model.forward(images_disp[1])
                if a1 is not None:
                    attn1, __ = visualize_attn(I_val, a1, up_factor=opt.base_up_factor, nrow=4)
                    writer.add_image('val/attention_map_1', attn1, epoch)
                if a2 is not None:
                    attn2, __ = visualize_attn(I_val, a2, up_factor=2*opt.base_up_factor, nrow=4)
                    writer.add_image('val/attention_map_2', attn2, epoch)

if __name__ == "__main__":
    if opt.preprocess:
        preprocess_data(root_dir='../data_2018')
    main()
