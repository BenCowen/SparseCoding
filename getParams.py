"""
need to put the header ;)

gets the params used to make pretty dictionaries for reproducibility's sake
"""


def fetchVizParams(datName):
    if datName == "MNIST10":
        batchSize = 10
        l1w = .3
        learnRate = 200
        LRDecay = 1

    elif datName == "CIFAR10":
        batchSize = 10
        l1w = 0.2
        learnRate = 50
        LRDecay   = 1

    elif datName == "FashionMNIST10":
        batchSize = 12
        l1w = 0.2
        learnRate = 1000
        LRDecay   = 0.99

    elif datName == "ASIRRA16":
        batchSize = 10
        l1w = 0.125
        learnRate = 250
        LRDecay = 1
 
    return batchSize, l1w, learnRate, LRDecay
