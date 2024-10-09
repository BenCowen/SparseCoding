#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom on-the-fly image transformations

@author: Benjamin Cowen, 7 Feb 2019
@contact: benjamin.cowen.math@gmail.com
"""

import torch
import torch.nn.functional as F

# MISC FCNS
def get_sparsity_ratio(codes):
    return codes.abs().gt(0).sum().detach().item() / codes.numel()

def add_frame(img, n_pix, fill_val=1):
    return F.pad(img, (n_pix, n_pix, n_pix, n_pix), value=fill_val)

# DEVICE
class AddToDevice(object):
    """ Adds sample to specified device (e.g. cuda:0 or cpu)."""

    def __init__(self, device):
        self.device = device

    def __call__(self, x):
        return x.to(self.device, non_blocking=True)


# PATCHITIZE
class OverlappingPatches(object):
    """
    Extracts **OVERLAPPING** patches from each image. Leaves color channel intact.
    """

    def __init__(self, config):
        """
        patch_size: number of pixels in one side of square patch
                     (square patches only)
        Reconstructable patch windowing inspired from:
            https://discuss.pytorch.org/t/fold-and-unfold-how-do-i-put-this-image-tensor-back-together-again/97374/3
        """
        self.psz = config['patch-size']
        self.stride = int(config['overlap-percentage'] * self.psz)
        # win = torch.signal.windows.cosine(self.psz)
        # self.win_patches = win.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        if config['vectorize']:
            self.vectorize = torch.nn.Flatten(start_dim=-2, end_dim=-1)
        else:
            self.vectorize = torch.nn.Identity()

    def __call__(self, img):
        # Pad image to be a multiple of the patch size and unfold into
        #   (batch x channel x sqrt{n-patch}?? x sqrt{n-patch}?? x psz x psz):
        # Note: using negative indices doesn't work as with indexing on unfold...
        col_dim = img.ndim-1
        row_dim = img.ndim-2
        pad_col = img.shape[col_dim] % self.psz // 2
        pad_row = img.shape[row_dim] % self.psz // 2
        patches = F.pad(img, (pad_col, pad_col,
                              pad_row, pad_row,
                              0, 0)).unfold(col_dim, self.psz, self.stride) \
                                    .unfold(row_dim, self.psz, self.stride).contiguous()
        # \
        # (patches - torch.mean(patches, (4, 5), keepdim=True)) \
        # .repeat(1, 1, 1, 1, self.psz, self.psz)
        # Window and return:
        # self.win_patches.repeat(1, 1, patches.shape[2], patches.shape[3], 1, 1).contiguous() * patches
        return self.vectorize(patches.transpose(-2, -1))


    def fold(self, vectorized_patches, img_shape):
        """
        Special thanks to these:
            https://discuss.pytorch.org/t/fold-and-unfold-how-do-i-put-this-image-tensor-back-together-again/97374/3
            https://discuss.pytorch.org/t/how-to-fold-the-unfolded/108162
        TODO: Need to do something with windowing/weighting here?
        TODO: How account for variable shapes? Seems impossible ...
        """
        # Reshape for Fold:
        bsz = vectorized_patches.shape[0]
        nc = vectorized_patches.shape[1]
        # Combine patch dimensions:
        batch_patch_per_channel = vectorized_patches.view(bsz, nc, -1, self.psz*self.psz
                                                          ).permute(1, 0, 3, 2).contiguous()
        # Do color channels separately:
        pp5 = []
        for channel in range(nc):
            pp5.append(F.fold(batch_patch_per_channel[channel], output_size=img_shape, kernel_size=self.psz, stride=self.stride))
        images = torch.cat(pp5, 1)
        return images

    def reconstruct(self, vectorized_patches, img_shape):
        """Re-fold patches and normalize them for perfect reconstruction"""
        unnorm_imgs = self.fold(vectorized_patches, img_shape)
        return unnorm_imgs / self.get_normalizer(unnorm_imgs.shape,
                                                 unnorm_imgs.device)

    def get_normalizer(self, full_shape, device='cpu'):
        att_name = f'normalizer_{full_shape}'.replace(".", "_")
        if not hasattr(self, att_name):
            setattr(self, att_name, self.fold(self(
                                          torch.ones(full_shape)),
                                          full_shape[-2:]).to(device))
        return getattr(self, att_name)
