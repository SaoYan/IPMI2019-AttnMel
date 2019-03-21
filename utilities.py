import numpy as np
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

def visualize_attn(I, a, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)

def compute_metrics(result_file, gt_file, threshold=0.5):
    # groundtruth
    with open(gt_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        gt = [int(row[2]) for row in reader]
    ##### prediction (probability) #####
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = list(map(float, row))
            pred.append(np.float32(prob[1]))
    # average precision
    AP = average_precision_score(gt, pred, average='macro')
    # area under ROC curve
    AUC = roc_auc_score(gt, pred)
    ##### prediction (binary) #####
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = list(map(float, row))
            pred.append(np.float32(prob[1] >= threshold))
    # precision & recall
    precision_mean = precision_score(gt, pred, average='macro')
    precision_mel  = precision_score(gt, pred, average='binary', pos_label=1)
    # recall
    recall_mean = recall_score(gt, pred, average='macro')
    recall_mel  = recall_score(gt, pred, average='binary', pos_label=1)
    return [AP, AUC, precision_mean, precision_mel, recall_mean, recall_mel]
