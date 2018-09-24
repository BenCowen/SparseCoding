#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load a single dataset 
all images are scaled to [0,1].
        
@author: Benjamin Cowen, Feb 24 2018
@contact: bc1947@nyu.edu, bencowen.com
"""

# TORCH
import os
import torch
import torchvision
import torchvision.transforms as transforms
import math as math

class scaleTo01(object):
    """
    scales one image to [0,1]
    """
    def __init__(self):
        pass
    def __call__(self, img):
        # set to [0, img.max()]
        new_im = img - img.min()
        maxVal = new_im.abs().max()
        if maxVal != 0:
            new_im = new_im/maxVal
        return new_im


# PATCHITIZE
class get_patches(object):
    """
    Extracts **NON-OVERLAPPING** patches from each image.
    """
    #TODO: it's wayyyy faster if we don't have to do this *EVERY TIME* an image
    #      is called. Should just do this once-and-for-all then put it in
    #      a new "patchitized" data loader....?

    def __init__(self,patch_size):
        """
        patch_size: number of pixels in one side of square patch
                     (square patches only)
        """
        #TODO add argument hv_cutoff: discard patches with pixel variance below this threshold
        #            (not implemented yet 3/15/2018)....

        self.patch_size = patch_size
        
    def __call__(self, img): #, variance_cutoff=float('inf')
        """
        img: image tensor of size (D,m,n)
        """
        pSz = self.patch_size
        D  = img.size(0)          # channels
        m  = img.size(1)          # rows
        n  = img.size(2)          # columns

        # If imsize = patchSize: DONE
        if m==pSz and m==n:
            return img

        # Otherwise, setup output tensor:
        Np  =  math.floor(m/pSz) # number of patches per dimension
        ppI = Np**2                     # patches per image

        if D==0:
            patches = torch.Tensor(ppI, pSz, pSz)
        else:
            patches = torch.Tensor(ppI, D, pSz, pSz)
            
        # Now, extract all the (non-overlapping!!!!!!!!!) patches
        next_patch = -1
        for i in range(Np):
            for j in range(Np):
                next_patch += 1
                cSel = i*pSz  # column select
                rSel = j*pSz  # row select
                patches[next_patch]= (img.narrow(1,rSel,pSz).narrow(2,cSel,pSz)).clone()
        return patches
    

# VECTORIZE
class vectorizeIm(object):
    """
    Vectorizes all patches in one image.
    """
    def __init__(self):
        pass
    def __call__(self, img):
        if img.dim()==3:  # regular (works WITHOUT using "get_patches")
            m = img.size(1)
            n = img.size(2)
            return img.resize_(m*n)
        elif img.dim()==4: # patchitized images
            Np = img.size(0)
            m = img.size(2)
            n = img.size(3)
            return img.resize_(Np,m*n)
            
def fixBsz(bsz,nppi):
    """
    Fix batchsize w.r.t. the number of PATCHES per image
    """
    if bsz<1:
        bsz = 1
    return bsz

##################################################################
## LOAD DATA (uses above classes)
##################################################################
def loadData(datName, patchSize, batchSize):
    """
    Loads dataset.
    datName must be "MNIST", "FashionMNIST", "CIFAR10", or "ASIRRA".
    """

    # Build preprocessing classes
    normalize  = scaleTo01()
    vectorize  = vectorizeIm()
    patchitize = get_patches(patchSize)

##################################################################
##################################################################
##################################################################
##################################################################
    if datName == 'MNIST':
        datPath = './DATA/MNIST'
    # Normalize,separate into patches
        m    = 32
        nppi = math.floor(m/patchSize)**2
        bsz  = fixBsz(batchSize,nppi )
        transform = transforms.Compose([
                     transforms.Resize((m,m)),   # not sure why it's 28x28 in pytorch
                     transforms.ToTensor(),
                     normalize,
                     patchitize,
                     vectorize])
#    # TRAINING SET
        trainset = torchvision.datasets.MNIST(root=datPath, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsz,
                                                  shuffle=True)
    # TESTING SET
        testset = torchvision.datasets.MNIST(root=datPath, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=bsz,
                                             shuffle=False)
##################################################################
##################################################################
##################################################################
##################################################################
    elif datName == 'FashionMNIST':
        datPath = './DATA/FashionMNIST'
    # Normalize, scale to the right size(zero pad?), separate into patches
        m    = 28
        nppi = math.floor(m/patchSize)**2
        bsz  = fixBsz(batchSize,nppi )
        transform = transforms.Compose(
                    [transforms.ToTensor(),
                     normalize,
                     patchitize,
                     vectorize])
#    # TRAINING SET
        trainset = torchvision.datasets.FashionMNIST(root=datPath, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsz,
                                                  shuffle=True)
    # TESTING SET    
        testset = torchvision.datasets.FashionMNIST(root=datPath, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,batch_size=bsz,
                                             shuffle=False)
##################################################################
##################################################################
##################################################################
##################################################################
    elif datName == 'CIFAR10':
        datPath = './DATA/CIFAR10'
    # Normalize, greyscale, separate into patches
        m    = 32
        nppi = math.floor(m/patchSize)**2
        bsz  = fixBsz(batchSize,nppi )
        transform = transforms.Compose([
                     transforms.Grayscale(),
                     transforms.ToTensor(),
                     normalize,
                     patchitize,
                     vectorize])
#    # TRAINING SET
        trainset = torchvision.datasets.CIFAR10(root=datPath, train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsz,
                                                  shuffle=True)
    # TESTING SET
        testset = torchvision.datasets.CIFAR10(root=datPath, train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=bsz,
                                             shuffle=False)
##################################################################
##################################################################
##################################################################
##################################################################
    elif datName == 'ASIRRA':
        traindir = 'DATA/ASIRRA/train'
        testdir  = 'DATA/ASIRRA/test'
    # Normalize, greyscale, separate into patches,vectorize,resize
        m    = 224
        nppi = math.floor(m/patchSize)**2
        bsz  = fixBsz(batchSize,nppi )
        transform = transforms.Compose([
                        #transforms.RandomResizedCrop(224),
                        transforms.Resize((m,m)),
                        #transforms.RandomHorizontalFlip(),
                        transforms.Grayscale(),
                        transforms.ToTensor(),
                        normalize,
                        patchitize,
                        vectorize])
     
        # TRAINING SET
        trainloader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(traindir, transform=transform),
            batch_size=bsz,
            shuffle=True,
            #pin_memory=True
            )
    
        # TESTING SET    
        testloader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(testdir, transform=transform),
            batch_size=bsz,
            shuffle= False,
            #pin_memory=False
            )
##################################################################
##################################################################
##################################################################
##################################################################
    else:
        ValueError('"dataset" MUST BE "MNIST", "FashionMNIST", "CIFAR10", OR "ASIRRA"')
##################################################################
##################################################################
##################################################################
##################################################################
    return trainloader,testloader


