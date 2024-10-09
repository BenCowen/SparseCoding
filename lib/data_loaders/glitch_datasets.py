"""
Dataset and Dataloader class for using Inaccessible Worlds SQL
 database with PyTorch.
"""
import os
import torch
import numpy as np
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import lib.dataset_generators.sql_tools as sql_tools
from non_committed_functions import get_data_root, get_datastats_path
import lib.UTILS.image_transforms as custom_transforms
import math


class GlitchDataLoader:
    def __init__(self, config):
        super(GlitchDataLoader, self).__init__()
        self.dataset = GlitchDataset(config)
        self.config = config
        self.train_loader = torch.utils.data.DataLoader(self.dataset,
                                                        config['batch-size'],
                                                        shuffle=True)
        self.valid_loader = self.train_loader

        # Set the data size:
        self.data_dim = self.get_data_dim(config)

    def __len__(self):
        return math.ceil(len(self.dataset) / self.config['batch-size'])

    def get_data_dim(self, config):
        if 'OverlappingPatches' in config['post-load-transforms']:
            resize_config = config['post-load-transforms']['OverlappingPatches']
            if resize_config['vectorize']:
                data_dim = resize_config['patch-size'] ** 2
            else:
                data_dim = (resize_config['patch-size'], resize_config['patch-size'])
        elif 'image-size' in config:
            data_dim = (config['image-size'], config['image-size'])
        return data_dim


class GlitchDataset(torch.utils.data.Dataset):
    """
    Reader that holds all of the data.
    """

    def __init__(self, config={}):

        # Save parameters
        self.batch_size = config['batch-size']

        # Get the available data:
        database = sql_tools.DataBase('inaccessible_worlds')
        x = np.array(database.execute_query('SELECT * FROM images;',
                                            need_fetch=True))
        database.close()

        self.data = {'ID': np.array([int(n) for n in x[:, 0]]),
                     'filepath': x[:, 1],
                     'bkgd_ratio': np.array([float(n) for n in x[:, 2]]),
                     'grayscale_var': np.array([float(n) for n in x[:, 3]]),
                     'nrow': np.array([int(n) for n in x[:, 4]]),
                     'ncol': np.array([int(n) for n in x[:, 5]])}

        self.stats = torch.load(os.path.join(get_datastats_path('inaccessible worlds')))
        self.add_chance = config['random-add']
        self.transform = self.setup_transforms(config['load-transforms'])
        self.N = len(self)

    def __len__(self):
        return len(self.data['ID'])

    def get_random_filename(self):
        return self.data['filepath'][torch.randint(low=0, high=self.N, size=(1, 1))]

    def __getitem__(self, idx):
        """ Returns one ready-to-go sample """
        filepath = self.data['filepath'][idx]
        x = self.transform(Image.open(filepath))
        if torch.randn(1) <= self.add_chance:
            random_file = self.get_random_filename()
            x += self.transform(Image.open(random_file))
        return x

    # TODO: put these in a super class:
    def setup_transforms(self, transform_specification):
        """ Initialize load-wise transforms"""
        if transform_specification is None:
            return custom_transforms.Identity()

        transform_list = []
        for transform_name, args in transform_specification.items():
            # Special cases
            if transform_name == 'Normalize':
                # Get pre-computer statistics
                transform_list.append(
                    torchvision.transforms.Normalize(
                        self.stats['mean'], self.stats['var']
                    ))
            elif transform_name == "Resize":
                # Ensure square
                transform_list.append(
                    getattr(torchvision.transforms, transform_name)(
                        size=(args['size'], args['size']))
                )
            else:
                transform_list.append(getattr(torchvision.transforms, transform_name)(**args))

        return torchvision.transforms.Compose(transform_list)


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
