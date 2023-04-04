"""
Custom loss functions

@author Benjamin Cowen
@date 3 April 2023
@contact benjamin.cowen.math@gmail.com
"""
import torch
import torchmetrics
from lib.UTILS.image_transforms import OverlappingPatches


class SSIM(torch.nn.Module):
    """ Reconstruct patches into imagery and THEN do SSIM."""

    def __init__(self, dataloader, **kwargs):
        super(SSIM, self).__init__()
        self.SSIM = torchmetrics.StructuralSimilarityIndexMeasure(**kwargs).to(dataloader.config['device'])

    def forward(self, img_true, img_est):
        return self.SSIM(img_true, img_est) / img_true.shape[0]


class L1(torch.nn.Module):
    """ Like seriously PyTorch?..."""

    def __init__(self,AAAHHH, **kwargs):
        super(L1, self).__init__()

    def forward(self, x):
        return x.abs().sum() / x.shape[0]
