#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Classes
        
@author: Benjamin Cowen, Feb 24 2018
@contact: bc1947@nyu.edu, bencowen.com/lsalsa
"""

class dataset1D(Dataset):
    """
    Class for single-domain data
    (for dict training for 1-dict sparse coding)
    """
    def __init__(self, root_dir,data_file, label_file, transform=None):
        """
        Args:
            binary_file (string): Path to the binary file with images (etc).
            label_file (string): Path to the binary file with labels (etc).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
















