#
# Copyright (C) 2024, TRASE
# Technical University of Munich CVG
# All rights reserved.
#
# TRASE is heavily based on other research. Consider citing their works as well.
# 3D Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting
# Deformable-3D-Gaussians: https://github.com/ingra14m/Deformable-3D-Gaussians
# gaussian-grouping: https://github.com/lkeab/gaussian-grouping
# SAGA: https://github.com/Jumpat/SegAnyGAussians
# SC-GS: https://github.com/yihua7/SC-GS
# 4d-gaussian-splatting: https://github.com/fudan-zvg/4d-gaussian-splatting
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting
# GRAPHDECO research group, https://team.inria.fr/graphdeco
#



import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def masked_l1_loss(network_output, gt, mask):
    mask = mask.float()[None,:,:].repeat(gt.shape[0],1,1)
    loss = torch.abs((network_output - gt)) * mask
    loss = loss.sum() / mask.sum()
    return loss

def weighted_l1_loss(network_output, gt, weight):
    loss = torch.abs((network_output - gt)) * weight
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def pixel_mask_correspondence_loss_positive(C, C_F, positive_th=0.75, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)

    positive_mask = torch.any(C == 1, dim = 0)
    positive_mask = torch.logical_and(positive_mask, ~diag_mask)
    positive_mask = torch.triu(positive_mask, diagonal=0) ## set the symmetric part to false
    number_of_all_pixel_pair = torch.nonzero(positive_mask).shape[0]
    positive_mask = torch.logical_and(positive_mask, C == 1)
    
    positive_mask = positive_mask.bool()
        
    if weights is not None:
        return (-weights[positive_mask]* C_F[positive_mask]).sum() / number_of_all_pixel_pair
    else:
        return (-C_F[positive_mask]).sum() / number_of_all_pixel_pair

def pixel_mask_correspondence_loss_negative(C, C_F, negative_th=0.5, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)
    negative_mask = torch.any(C == 0, dim = 0)
    negative_mask = torch.logical_and(negative_mask, ~diag_mask)
    negative_mask = torch.triu(negative_mask, diagonal=0) ## set the symmetric part to false
    number_of_all_pixel_pair = torch.nonzero(negative_mask).shape[0]
    negative_mask = torch.logical_and(negative_mask, C == 0)
    negative_mask = negative_mask.bool()
        
    if weights is not None:
        return (weights[negative_mask] * torch.relu(C_F[negative_mask])).sum() / number_of_all_pixel_pair
    else:
        return (torch.relu(C_F[negative_mask])).sum() / number_of_all_pixel_pair

def pixel_mask_correspondence_loss_soft_hard_positive(C, C_F, positive_th=0.75, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)
    soft_hard_positive_mask = torch.any(torch.logical_and(C_F < positive_th, C == 1), dim = 0)
    soft_hard_positive_mask = torch.logical_and(soft_hard_positive_mask, ~diag_mask)
    soft_hard_positive_mask = torch.triu(soft_hard_positive_mask, diagonal=0) ## set the symmetric part to false
    
    number_of_all_pixel_pair = torch.nonzero(soft_hard_positive_mask).shape[0]
    soft_hard_positive_mask = torch.logical_and(soft_hard_positive_mask, C == 1)
    
    soft_hard_positive_mask = soft_hard_positive_mask.bool()
    
    if soft_hard_positive_mask.sum() == 0: ## No positvie sample found
        print("[WARNING] no positive sample found")
        return 0.0
    else:
        if weights is not None:
            loss = (-weights[soft_hard_positive_mask] * C_F[soft_hard_positive_mask]).sum() / number_of_all_pixel_pair
            
        else:
            loss = (-C_F[soft_hard_positive_mask]).sum() / number_of_all_pixel_pair
            
        return loss

def pixel_mask_correspondence_loss_soft_negative(C, C_F, negative_th=0.5, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C_F.shape[0], dtype=bool, device=C_F.device)
    soft_hard_negative_mask = torch.any(torch.logical_and(C_F > negative_th, C == 0), dim = 0)
    
    soft_hard_negative_mask = torch.logical_and(soft_hard_negative_mask, ~diag_mask)
    
    soft_hard_negative_mask = torch.triu(soft_hard_negative_mask, diagonal=0) ## set the symmetric part to false
    
    number_of_all_pixel_pair = torch.nonzero(soft_hard_negative_mask).shape[0]
    
    soft_hard_negative_mask = torch.logical_and(soft_hard_negative_mask, C == 0)
    soft_hard_negative_mask = soft_hard_negative_mask.bool()
        
    if soft_hard_negative_mask.sum() == 0:
        print("[WARNING] no negative sample found")
        return 0.0
    else:
        if weights is not None:
            loss = (weights[soft_hard_negative_mask] * torch.relu(C_F[soft_hard_negative_mask])).sum() / number_of_all_pixel_pair
        else:
            loss = (torch.relu(C_F[soft_hard_negative_mask])).sum() / number_of_all_pixel_pair

        return loss

def pixel_mask_correspondence_loss_hard_positive(C, C_F, positive_th=0.75, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C.shape[0], dtype=bool, device=C_F.device)

    # Find hard positive indices (i, j)
    hard_positive_mask = torch.triu((C_F < positive_th) & (C == 1) & (~diag_mask), diagonal=0)
    hard_positive_indices = torch.nonzero(hard_positive_mask, as_tuple=False)

    if hard_positive_indices.shape[0] == 0:
        print("[WARNING] no hard positive sample found")
        return torch.tensor(0.0, device=C_F.device)

    i, j = hard_positive_indices[:, 0], hard_positive_indices[:, 1]

    # Compute loss
    C_F_hard = C_F[i, j]  # Use indexed values instead of masked tensor

    if weights is not None:
        loss = (-weights[i, j] * C_F_hard).mean()
    else:
        loss = (-C_F_hard).mean()

    return loss

def pixel_mask_correspondence_loss_hard_negative(C, C_F, negative_th=0.5, weights=None, verbose=False, log_tb=False, tb_writer=None, iteration=None):
    diag_mask = torch.eye(C.shape[0], dtype=bool, device=C_F.device)

    # Find hard negative indices (i, j)
    hard_negative_mask = torch.triu((C_F > negative_th) & (C == 0) & (~diag_mask), diagonal=0)
    hard_negative_indices = torch.nonzero(hard_negative_mask, as_tuple=False)
    if hard_negative_indices.shape[0] == 0:
        print("[WARNING] no hard negative sample found")
        return torch.tensor(0.0, device=C_F.device)

    i, j = hard_negative_indices[:, 0], hard_negative_indices[:, 1]

    # Compute loss
    C_F_hard = C_F[i, j]

    if weights is not None:
        loss = (weights[i, j] * torch.relu(C_F_hard)).mean()
    else:
        loss = (torch.relu(C_F_hard)).mean()

    return loss
    
positive_pixel_pair_loss = {
    'hard': pixel_mask_correspondence_loss_hard_positive,
    'all': pixel_mask_correspondence_loss_positive,
    'soft': pixel_mask_correspondence_loss_soft_hard_positive
}

negative_pixel_pair_loss = {
    'hard': pixel_mask_correspondence_loss_hard_negative,
    'all': pixel_mask_correspondence_loss_negative,
    'soft': pixel_mask_correspondence_loss_soft_negative
}