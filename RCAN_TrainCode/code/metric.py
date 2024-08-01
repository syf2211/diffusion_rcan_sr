import math
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
from torch import nn
import cv2

def ssim(image1, image2, cuda, K=(0.01, 0.03), window_size=11, L=1):
    _, channel1, _, _ = image1.size()
    _, channel2, _, _ = image2.size()
    channel = min(channel1, channel2)
    image1 = image1.to(cuda)
    image2 = image2.to(cuda)
    # Gaussian window generation
    sigma = 1.5  # default
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    window = window.to(cuda)

    # Define constants
    C1 = (K[0]*L) ** 2
    C2 = (K[1]*L) ** 2

    mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 * image1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(image2 * image2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1 
    print(np.max(img1), np.min(img1), np.max(img2), np.min(img2))
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

#20 * math.log10(1 / math.sqrt(np.mean((gt - sr) ** 2)))
# SSIM(gt, sr, data_range=1)