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

import pandas as pd
import os


class WesadDataloader:

    def __init__(self, subject_list, dataset_basedir):

        self.datadir = dataset_basedir
        subject_num = subject_list[0]
        data_dicts = pd.read_pickle(self.make_S_filename(subject_num))
        wrist_acc = data_dicts["signal"]["wrist"]["ACC"]
        wrist_bvp = data_dicts["signal"]["wrist"]["BVP"]
        wrist_eda = data_dicts["signal"]["wrist"]["EDA"]
        wrist_temp = data_dicts["signal"]["wrist"]["TEMP"]
        lbl = data_dicts["label"].reshape(data_dicts["label"].shape[0], 1)
        4

    def make_S_filename(self, subject_num):
        sn = 'S{}'.format(subject_num)
        return os.path.join(self.datadir, sn, '{}.pkl'.format(sn))

if __name__ == "__main__":
    base_data_dir = 'K:\\DATASETS\\WESAD'
    train_set = [2]
    f = WesadDataloader(train_set, base_data_dir)