#!/usr/bin/env python
# coding=utf-8
from librarys import *
# from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
def dice_loss(pred, target, smooth=1.):
    # pred = pred.contiguous()
    # target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) /
                 (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=.5):
    pred = F.sigmoid(pred)
    class_weight = torch.FloatTensor([1, 613])  # TODO 1:10?
    bce = F.binary_cross_entropy(
        pred, target, weight=class_weight[target.long()].to(device))

    # pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['L_bce'] += bce.detach().item() * target.size(0)
    metrics['L_dice'] += dice.detach().item() * target.size(0)
    metrics['loss'] += loss.detach().item() * target.size(0)

    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n



def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


# class StableBCELoss(torch.nn.modules.Module):
#     def __init__(self):
#         super(StableBCELoss, self).__init__()
#
#     def forward(self, input, target):
#         neg_abs = - input.abs()
#         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
#         return loss.mean()


# def binary_xloss(logits, labels, ignore=None):
#     """
#     Binary Cross entropy loss
#       logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
#       labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
#       ignore: void class id
#     """
#     logits, labels = flatten_binary_scores(logits, labels, ignore)
#     loss = StableBCELoss()(logits, Variable(labels.float()))
#     return loss


# def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
#     """
#     Array of IoU for each (non ignored) class
#     """
#     if not per_image:
#         preds, labels = (preds,), (labels,)
#     ious = []
#     for pred, label in zip(preds, labels):
#         iou = []
#         for i in range(C):
#             # The ignored label is sometimes among predicted classes (ENet - CityScapes)
#             if i != ignore:
#                 intersection = ((label == i) & (pred == i)).sum()
#                 union = ((label == i) | (
#                     (pred == i) & (label != ignore))).sum()
#                 if not union:
#                     iou.append(EMPTY)
#                 else:
#                     iou.append(float(intersection) / float(union))
#         ious.append(iou)
#     # mean accross images if per_image
#     ious = [mean(iou) for iou in zip(*ious)]
#     return 100 * np.array(ious)


# def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
#     """
#     IoU for foreground class
#     binary: 1 foreground, 0 background
#     """
#     if not per_image:
#         preds, labels = (preds,), (labels,)
#     ious = []
#     for pred, label in zip(preds, labels):
#         intersection = ((label == 1) & (pred == 1)).sum()
#         union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
#         if not union:
#             iou = EMPTY
#         else:
#             iou = float(intersection) / float(union)
#         ious.append(iou)
#     iou = mean(ious)    # mean accross images if per_image
#     return 100 * iou

#
# class FocalLoss2d(nn.Module):
#
#     def __init__(self, gamma=2, size_average=True):
#         super(FocalLoss2d, self).__init__()
#         self.gamma = gamma
#         self.size_average = size_average
#
#     def forward(self, logit, target, class_weight=None, type='softmax'):
#         target = target.view(-1, 1).long()
#
#         if type == 'sigmoid':
#             if class_weight is None:
#                 class_weight = [1]*2  # [0.5, 0.5]
#
#             prob = F.sigmoid(logit)
#             prob = prob.view(-1, 1)
#             prob = torch.cat((1-prob, prob), 1)
#             select = torch.FloatTensor(len(prob), 2).zero_().cuda()
#             select.scatter_(1, target, 1.)
#
#         elif type == 'softmax':
#             B, C, H, W = logit.size()
#             if class_weight is None:
#                 class_weight = [1]*C  # [1/C]*C
#
#             logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
#             prob = F.softmax(logit, 1)
#             select = torch.FloatTensor(len(prob), C).zero_().cuda()
#             select.scatter_(1, target, 1.)
#
#         class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
#         class_weight = torch.gather(class_weight, 0, target)
#
#         prob = (prob*select).sum(1).view(-1, 1)
#         prob = torch.clamp(prob, 1e-8, 1-1e-8)
#         batch_loss = - class_weight * \
#             (torch.pow((1-prob), self.gamma))*prob.log()
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss
#
#         return loss

#
# # http://geek.csdn.net/news/detail/126833
# class PseudoBCELoss2d(nn.Module):
#     def __init__(self):
#         super(PseudoBCELoss2d, self).__init__()
#
#     def forward(self, logit, truth):
#         z = logit.view(-1)
#         t = truth.view(-1)
#         loss = z.clamp(min=0) - z*t + torch.log(1 + torch.exp(-z.abs()))
#         loss = loss.sum()/len(t)  # w.sum()
#         return loss
#
# def lovasz_hinge(logits, labels, per_image=True, ignore=None):
#     """
#     Binary Lovasz hinge loss
#       logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
#       labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
#       per_image: compute the loss per image instead of per batch
#       ignore: void class id
#     """
#     if per_image:
#         loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
#                           for log, lab in zip(logits, labels))
#     else:
#         loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
#     return loss


def lovasz_hinge_relu(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat_relu(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat_relu(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted)+1, Variable(grad))
    return loss

def lovasz_hinge_flat_relu(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

