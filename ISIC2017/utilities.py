import numpy as np
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, precision_score, recall_score

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def visualize_attn_softmax(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    N,C,W,H = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = img + attn
    min_val, max_val = np.min(vis), np.max(vis)
    vis = (vis - min_val) / (max_val - min_val)
    return torch.from_numpy(vis).permute(2,0,1)

def visualize_attn_sigmoid(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    a = torch.sigmoid(c)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = img + attn
    min_val, max_val = np.min(vis), np.max(vis)
    vis = (vis - min_val) / (max_val - min_val)
    return torch.from_numpy(vis).permute(2,0,1)

def compute_metrics(result_file):
    # groundtruth
    with open('test.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        gt = [int(row[1]) for row in reader]
    # prediction
    pred = []
    i = 0
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = list(map(float, row))
            pred.append(prob[1])
            i += 1
    # compute mAP
    mAP = average_precision_score(gt, pred)
    # mAP = average_precision_score(gt, pred, average='samples')
    # compute precision and recall
    precision, recall, __ = precision_recall_curve(gt, pred)
    # compute AUC
    AUC = roc_auc_score(gt, pred)
    # plot ROC curve
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    ax.step(recall, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    canvas.draw()
    I = np.fromstring(canvas.tostring_rgb(), dtype='uint8', sep='')
    I = I.reshape(canvas.get_width_height()[::-1]+(3,))
    I = np.transpose(I, [2,0,1])
    return mAP, AUC, torch.Tensor(np.float32(I))

def compute_mean_pecision_recall(result_file):
    # groundtruth
    with open('test.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        gt = [int(row[1]) for row in reader]
    # prediction
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = np.array(list(map(float, row)))
            pred.append(np.argmax(prob))
    # compute precision & recall
    precision  = precision_score(gt, pred, average='macro')
    recall     = recall_score(gt, pred, average='macro')
    recall_mel = recall_score(gt, pred, average='binary', pos_label=1)
    return precision, recall, recall_mel

def returnCAM(I, feature_conv, weight_softmax, class_idx, im_size, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    weights = weight_softmax[class_idx,:]
    maps = []
    for feature in feature_conv:
        N, C, H, W = feature.shape
        cam = np.einsum('j,ijk->ik', weights, feature.reshape((N,C,W*H))).reshape(N,W,H)
        cam = np.expand_dims(cam, 1)
        maps.append(cam)
    cam = np.concatenate(maps, axis=0)
    cam = F.interpolate(torch.from_numpy(cam), size=(im_size,im_size), mode='bilinear', align_corners=False)
    cam = utils.make_grid(cam, nrow=nrow, normalize=True, scale_each=True)
    cam = cam.permute((1,2,0)).mul(255).byte().cpu().numpy()
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    cam = np.float32(cam) / 255
    vis = 0.7 * img + 0.3 * cam
    return torch.from_numpy(vis).permute(2,0,1)
