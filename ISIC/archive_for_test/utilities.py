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
    vis = 0.4 * img + 0.6 * attn
    # vis = attn
    return torch.from_numpy(vis).permute(2,0,1), a, [torch.max(a).item(),torch.min(a).item(),torch.mean(a).item()]

def compute_metrics(result_file):
    # groundtruth
    with open('test.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        gt = [int(row[2]) for row in reader]
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
    mAP = average_precision_score(gt, pred, average='macro')
    # compute AUC
    AUC = roc_auc_score(gt, pred)
    # plot ROC curve
    precision, recall, __ = precision_recall_curve(gt, pred, pos_label=1)
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

def compute_mean_pecision_recall(result_file, threshold=0.5):
    # groundtruth
    with open('test.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        gt = [int(row[2]) for row in reader]
    # prediction
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = np.array(list(map(float, row)))
            pred.append(np.float32(prob[1] >= threshold))
    # compute precision & recall
    precision  = precision_score(gt, pred, average='macro')
    recall     = recall_score(gt, pred, average='macro')
    precision_mel  = precision_score(gt, pred, average='binary', pos_label=1)
    recall_mel = recall_score(gt, pred, average='binary', pos_label=1)
    return precision, recall, precision_mel, recall_mel

def jaccard_similarity_coefficient(A, B, no_positives=1.0, error_check=False):
    """Returns the jaccard index/similarity coefficient between A and B.

    This should work for arrays of any dimensions.

    J = len(intersection(A,B)) / len(union(A,B))

    To extend to probabilistic input, to compute the intersection, use the min(A,B).
    To compute the union, use max(A,B).

    Assumes that a value of 1 indicates the positive values.
    A value of 0 indicates the negative values.

    If no positive values (1) in either A or B, then returns `no_positives`.

    """
    # Make sure the shapes are the same.
    if not A.shape == B.shape:
        raise ValueError("A and B must be the same shape")

    if error_check:
        # Make sure values are between 0 and 1.
        if np.any((A > 1.) | (A < 0) | (B > 1.) | (B < 0)):
            raise ValueError("A and B must be between 0 and 1")

    # Flatten to handle nd arrays.
    A = A.flatten()
    B = B.flatten()

    intersect = np.minimum(A, B)
    union = np.maximum(A, B)

    union_sum = union.sum()
    # Special case if neither A or B have a 1 value.
    if union_sum == 0:
        return no_positives

    # Compute the Jaccard.
    J = float(intersect.sum()) / union_sum
    return J

'''
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
'''
