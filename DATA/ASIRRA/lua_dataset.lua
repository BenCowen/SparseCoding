require 'dataset/imageset'
require 'image'
--[[
catORdog = 'cats'
gray == 'False'
size = 224

d = ImageSet.dataset({dir = './data/train/'..catORdog})

if gray == true then
  X_name = catORdog..'_gray_dataset'
  full_dataset = torch.Tensor(1000, 1,size,size)
else
  X_name = catORdog..'_dataset'
  full_dataset = torch.Tensor(1000, 3,size,size)
end



for i = 1, 1000 do
  A = image.scale(d.dataset.data[i], size,size, 'bicubic')
  full_dataset:select(1,i):copy(A)
end


images = image.toDisplayTensor({input = full_dataset:narrow(1,1,64), nrow = 8, padding = 1})
image.save(X_name..'_'..size..'.png', images)

dataa = {}
dataa.data = full_dataset

if gray == true then
   torch.save(catORdog..'_gray_trainx1x'..size..'x'..size..'.t7', dataa)
else
   torch.save(catORdog..'_trainx3x'..size..'x'..size..'.t7', dataa)
end
]]
require 'cutorch'
require 'nn'
require 'cunn'

torch.manualSeed(0)

size = 224

datasize = 500  -- testdata = 500
    -- traindata = 2000
datasetmode = 'valid' --train or valid

c = ImageSet.dataset({dir = './cats_and_dogs_new/'..datasetmode..'/cats'})
d = ImageSet.dataset({dir = './cats_and_dogs_new/'..datasetmode..'/dogs'})


AA = c.dataset.data
BB = d.dataset.data
c = nil
d = nil


full_dataset = torch.Tensor(2*datasize,3,size, size):cuda()
labels = torch.Tensor(2*datasize)
A = torch.CudaTensor(3,size, size)
B = torch.CudaTensor(3, size, size)
for i = 1,datasize do
  print(i)
  A:copy(image.scale(AA[i], size, size, 'bicubic'))
  B:copy(image.scale(BB[i], size, size, 'bicubic'))
  full_dataset:select(1, 2*(i-1) + 1):copy(A)
  labels[2*(i-1)+1] = 0--:select(1, 2*(i-1)+1):fill(0)
  full_dataset:select(1, 2*i):copy(B)
  labels[2*i] = 1--:select(1,2*i):fill(1)
end
AA = nil
BB = nil
A = nil
B = nil

new_full_dataset = torch.Tensor(full_dataset:size()):cuda()
--labels = torch.Tensor(2*datasize)
--labels:narrow(1,1,datasize):fill(0)
--labels:narrow(1,datasize+1,datasize):fill(1)
shuffled_labels = torch.Tensor(2*datasize)
shuffle = torch.randperm(2*datasize)

double_datasize = 2*datasize 

for j = 1, double_datasize do
  print(j)
  new_full_dataset:select(1,j):copy(full_dataset:select(1, shuffle[j]))
  shuffled_labels[j] = labels[shuffle[j]]
end


images = image.toDisplayTensor({input = new_full_dataset:narrow(1,1,64), nrow = 8, padding = 1})
image.save(datasetmode..'_mixed_size'..size..'.png', images)


dataa = {}
dataa.data = new_full_dataset:float()
dataa.labels = shuffled_labels
torch.save('cats_dogs_test_sz'..size..'.t7', dataa)






