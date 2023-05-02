"""
Dataset and Dataloader class for using Inaccessible Worlds SQL
 database with PyTorch.
"""
import os
import torch
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import lib.dataset_generators.sql_tools as sql_tools
from non_committed_functions import get_data_root, get_datastats_path
import lib.UTILS.image_transforms as custom_transforms

class GlitchDataset:
    """
    Reader that holds all of the data.
    """

    def __init__(self, transforms=[]):
        # Get the available data:
        database = sql_tools.DataBase('inaccessible_worlds')
        x = np.array(database.execute_query('SELECT * FROM images;'),
                     need_fetch=True)
        database.close()
        self.data = {'ID': np.array([int(n) for n in x[:, 0]]),
                     'filepath': x[:, 1],
                     'bkgd_ratio': np.array([float(n) for n in x[:, 2]]),
                     'grayscale_var': np.array([float(n) for n in x[:, 3]]),
                     'nrow': np.array([int(n) for n in x[:, 4]]),
                     'ncol': np.array([int(n) for n in x[:, 5]])}

        self.stats = torch.load(os.path.join(get_datastats_path('inaccessible_worlds'))

        self.transform = self.setup_transforms(transforms)

    def __len__(self):
        return len(self.data['ID'])

    def __getitem__(self, idx):
        filepaths = self.data['filepath'][idx]
        imgs = io.imread(filepaths)


    def setup_transforms(self, transform_specification):
        """ Initialize load-wise transforms"""
        if transform_specification is None:
            return custom_transforms.Identity()

        transform_list = []
        for transform in transform_specification:
            # Setup the transform if it has a keyword here
            transform_list.append(transform)

        return transforms.Compose(transform_list)


if __name__ == "__main__":
    stats_subdir = 'dataset-statistics'
    g = GlitchDataset()
    for trait, x_label, y_label in zip(['bkgd_ratio', 'grayscale_var'],
                                       ['% pixels equal to mode',
                                        'pixel variance (in grayscale)'],
                                       ['# samples with this ratio',
                                        '# samples with this variance']
                                       ):
        plt.clf()
        plt.hist(g.data[trait],
                 bins=4,
                 density=False,
                 color='blue')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid()
        plt.title(f'Inaccessible Worlds Statistics (N = {len(g)})')
        plt.savefig(os.path.join(get_data_root('inaccessible worlds'),
                                 stats_subdir,
                                 f'{trait}_hist.png'))
