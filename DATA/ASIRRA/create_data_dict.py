# Following advice from:
#     https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
# creating a dictionray that defines data partitions etc



import os
import random
import torch
from torch.utils import data

#from loadImDat import loadData
#from AUX.class_pArray import pArray

def defineTrainValSets(split = 1,
                       totalImsToUse = 4000,
                       RNGSEED=23):
    # dataset size
    #    split = 0.8
    #    totalImsToUse    = 4000
    totalImsPerClass = totalImsToUse/2
    trainCutOff      = round(split*totalImsPerClass)
    
    # RANDOM SEED (repeatability)
#    RNGSEED=23
    random.seed(RNGSEED)
    
    # WHERE TO FIND IMAGES (paths)
    trainPath = 'train/'
    animals = ['cats','dogs']
    
    # DICTIONARIES FOR LOADING
    trainLabels = {}
    valLabels   = {}
    trainList   = []
    valList     = []
    
    for label,animal in enumerate(animals):
        k=0
        
        # The following for-loop has only 1 iteration
        #    it's just a roundabout way of getting list of 
        #    all filed in trainPath+animal+'/'.
        z = animal+'/'
        for root, dirs, files in os.walk(trainPath+z):
            random.shuffle(files)
            for imageName in files:
                k+=1
                if k< trainCutOff:
                    trainList.append(z+imageName)
                    trainLabels[imageName]=label
                elif k< totalImsPerClass:
                    valList.append(z+imageName)
                    valLabels[imageName]=label
                else:
                    break
    
    partition = {'train':trainList, 'validation':valList}
    labels = {}
    labels.update(trainLabels)
    labels.update(valLabels)
    return partition,labels





class asirraDataSet(data.Dataset):
    
    def __init__(self,list_IDs, labels,trainPath):
        'Initialization'
        self.labels    = labels
        self.list_IDs  = list_IDs
        self.trainPath = trainPath
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        print(self.trainPath+ID)
        X = torch.load(self.trainPath + ID)
        y = self.labels[ID]

        return X, y


#def asirraDataLoader():
#    train_set = asirraDataSet( partition['train'], labels )
#    val_set   = asirraDataSet( partition['validation'], labels )
#    trainLoader = data.DataLoader(train_set,**params)
#    valLoader   = data.DataLoader(val_set,**params)

#class mixedMNISTdataset(data.Dataset):
#  def __init__(self, otherDataset,totalMixedIms):
#    self.otherDat = otherDataset
#    self.totalMixedIms
#    self.MNISTloader=

  #datArgs = pArray()
#  datArgs.batch_size = batch_size
#  datArgs.patch_size = 28
#  trainSet,testSet = loadData(dataset,datArgs)

#  def __len__(self):
#    return self.totalMixedIms

#  def __getitem__(self,index):
#    return 23
    # get one image from each dataset and return sum













