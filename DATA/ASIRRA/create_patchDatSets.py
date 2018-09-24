# loop through all the images and cut into patches.
# ugh...?

import os
import torch


patchsize = 28
loadWholeImPath = 'train/'
savePatchPath = 'ptrain'+str(patchsize)+'/'
animals   = ['cats','dogs']

########################
## FOR PATCHSIZE = 28
##    AND ASIRRA IMAGES [224,224]:
Np        = 8
k=0

for animal in animals:
  z = animal+'/'
  for root, dirs, files in os.walk(loadWholeImPath+z):
    for imageName in files:
      # load image
      I = torch.load(root+imageName)
      # chop into patches
      patchList = [23]*(8**2)
      next_patch = -1
      for i in range(Np):
        for j in range(Np):
          next_patch += 1
          cSel = i*pSz  # column select
          rSel = j*pSz  # row select
          patchList[next_patch]= (img.narrow(1,rSel,pSz).narrow(2,cSel,pSz)).clone()
  
      # save each patch in new directory
      for patch_ID,im_patch in enumerate(patchList):
        torch.save(im_patch, savePatchPath+z+im_ID+patch_ID)

      k+=1
      if k>0:
        break
