#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for nonlinearities

@author: Benjamin Cowen, Feb 24 2018
@contact: bc1947@nyu.edu, bencowen.com/lsalsa
"""
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

def nothing():
    pass

class soft_thresh(nn.Module):

    def __init__(self,lams,n_list=[],use_cuda=False):
        super(soft_thresh, self).__init__()
        
        self.use_cuda = use_cuda
        self.shrink = []
        self.n_list = n_list
        
        if type(lams)==float:
            self.lams = [lams]
            self.shrink.append( nn.Softshrink(lams) )
            
        elif type(lams)==list:
            self.lams = lams
            for lam in lams:
               self.shrink.append( nn.Softshrink(lam) ) 
               
               
        if use_cuda:
            for s,shr in enumerate(self.shrink):
               self.shrink[s]= shr.cuda()

#        soft_thresh.register_forward_pre_hook(nothing)
       
    def forward(self,input):

        if len(self.lams)==1:
           return self.shrink[0](input)
        else:
           outputs = []
           beg = 0
           n_d = 0
           for t,shrink in enumerate(self.shrink):
               beg += n_d
               n_d = self.n_list[t]
               outputs.append( shrink(input.narrow(1,beg,n_d)) )
           return torch.cat(outputs,1)
           

        
        
class one_hard_thresh(nn.Module):
    
    def __init__(self, lam, wantcuda):
        super(one_hard_thresh, self).__init__()
        if wantcuda:
            self.hard = nn.Threshold(lam,0).cuda()
        else:
            self.hard = nn.Threshold(lam,0)
        
    def forward(self,input):
        return torch.mul(torch.sign(input),self.hard(input.abs()))





class hard_thresh(nn.Module):

    def __init__(self,lams,n_list=[],use_cuda=False):
        super(hard_thresh, self).__init__()
        
        
        self.use_cuda = use_cuda
        self.thrsh = []
        self.n_list = n_list
        
        if type(lams)==float:
            self.lams = [lams]
            self.thrsh.append( one_hard_thresh(lams,use_cuda) )
            
        elif type(lams)==list:
            self.lams = lams
            for lam in lams:
               self.thrsh.append( one_hard_thresh(lam,use_cuda) ) 
               
               
        if use_cuda:
            for s,th in enumerate(self.thrsh):
               self.thrsh[s]= th.cuda()
               
#        hard_thresh.register_forward_pre_hook(nothing)
       
    def forward(self,input):

        if len(self.lams)==1:
           return self.thrsh[0](input)
        else:
           outputs = []
           beg = 0
           n_d = 0
           for t,thrsh in enumerate(self.thrsh):
               beg += n_d
               n_d = self.n_list[t]
               outputs.append( thrsh(input.narrow(1,beg,n_d)) )
           return torch.cat(outputs,1)
           
       
def makeSALSAthreshes(l1w,mu,ns,use_cuda):

        if type(l1w)==float:
            th = l1w/mu
        elif type(l1w)==list:
            th = []
            for a in l1w:
                th.append(1.0*a/mu)

        shrink = soft_thresh(th, ns, use_cuda)
        hard   = hard_thresh(th, ns, use_cuda)
        return shrink,hard





