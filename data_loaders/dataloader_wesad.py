#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataloader defining / extracting samples from WESAD.

TODO: make an dataloader abstract class for future reference, esp getting to MCA...
TODO list:
  1. Should be able to load all into RAM for now. but probably should make
        a prepickle mode or something.
  2. Give a master sampling freq, resample all to same (?)
        a.) make multi-channel visualizer
  3. Need to define what a sample is gonna be.
        a.) a chunk of time
  4. Try sparse-coding some
  5. learn dictionary; alternate decoder, encoder

@author: Benjamin Cowen, Feb 21 2023
@contact: benjamin.cowen.math@gmail.com
"""
import matplotlib.pyplot as plt
from pandas import read_pickle as pandas_read_pickle
import numpy as np
import torch
import os


class STFT(torch.nn.Module):

    def __init__(self, **kwargs):
        super(STFT, self).__init__()
        self.stft_dict = kwargs

    def forward(self, inputs):
        return torch.stft(inputs, **self.stft_dict)


class WesadDataloader:

    def __init__(self, subject_list, dataset_basedir, dtype=torch.float):
        self.dtype = dtype
        self.datadir = dataset_basedir
        subject_num = subject_list[0]
        data_dicts = pandas_read_pickle(self.make_S_filename(subject_num))
        self.acc = torch.Tensor(data_dicts["signal"]["wrist"]["ACC"]).type(self.dtype)
        self.bvp = torch.Tensor(data_dicts["signal"]["wrist"]["BVP"]).type(self.dtype)
        self.eda = torch.Tensor(data_dicts["signal"]["wrist"]["EDA"]).type(self.dtype)
        self.temp = torch.Tensor(data_dicts["signal"]["wrist"]["TEMP"]).type(self.dtype)
        self.label = torch.Tensor(data_dicts["label"].reshape(data_dicts["label"].shape[0], 1)).type(self.dtype)
        self.Fs = {'acc': 32, 'bvp': 64, 'eda': 4, 'temp': 4, 'label': 700}

    def make_S_filename(self, subject_num):
        sn = 'S{}'.format(subject_num)
        return os.path.join(self.datadir, sn, '{}.pkl'.format(sn))

    def get_window(self, window_type, window_len):
        if window_type == 'hann':
            return torch.hann_window(int(window_len))

    def setup_spectrogram(self, channel_name, win_len_sec=10,
                          window_type='hann', amount_overlap=0.25):
        """
        Setup STFT transform for a given channel
        """
        win_length_samples = self.Fs[channel_name] * win_len_sec
        fft_len = 2 ** int(np.ceil(np.log2(win_length_samples)))
        hop_length = int(np.floor(win_length_samples / amount_overlap))
        channel_stft_name = channel_name + '_stft'
        return_complex = self.dtype == torch.complex64
        setattr(self, channel_stft_name,
                STFT(n_fft=fft_len, hop_length=hop_length, win_length=win_length_samples,
                     window=self.get_window(window_type, win_length_samples),
                     return_complex=return_complex))

    def prep_stft(self, spectrogram, style='wide'):
        if style=='wide':
            rot = lambda x: x.transpose(0,1)
        else:
            rot = lambda x: x
        return np.log10(rot(spectrogram).abs().squeeze())


if __name__ == "__main__":
    base_data_dir = 'K:\\DATASETS\\WESAD'
    train_set = [2]

    f = WesadDataloader(train_set, base_data_dir, dtype=torch.cfloat)
    f.setup_spectrogram('acc')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,10))
    plt.imshow(f.prep_stft(f.acc_stft(f.acc[:, 0].view(1, -1))))
    plt.savefig('SP.png')
