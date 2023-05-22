"""
Config reader and wrapper for the whole system.

@author Benjamin Cowen
@date 3 April 2023
@contact benjamin.cowen.math@gmail.com
"""

import torch
import torchvision
import torchvision.transforms as transforms
import lib.UTILS.image_transforms as custom_transforms


class PyTorchDataset:
    """
    This just serves to generate/download a PyTorch dataloader. Thrown together from
            https://github.com/jpowie01/DCGAN_CelebA/blob/master/dataset.py
    See parse_config for args.
    """
    _default_img_size = 64
    _default_normalize_value = 0.5
    _default_target_list = ['attr', 'landmarks']
    _default_batch_size = 256
    _default_num_workers = 1

    def __init__(self, config={}):

        self.config = config
        self.parse_config(config)

        transform_list = []
        if 'image-size' in config:
            transform_list.append(transforms.Resize((config['image-size'],
                                                     config['image-size'])))

        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((self.norm_val, self.norm_val, self.norm_val),
                                                   (self.norm_val, self.norm_val, self.norm_val)))

        # TODO: lmao has to be a better way here...
        # if 'custom-transforms' in config:
        #     for t_name, t_config in config['custom-transforms'].items():
        #         transform_list.append(getattr(custom_transforms, t_name)(t_config))

        # transform_list.append(custom_transforms.AddToDevice(device=config['device']))
        transform_list = transforms.Compose(transform_list)

        train_dataset = torchvision.datasets.ImageFolder(
            self.data_dir,
            transform=transform_list)

        # Use sampler for randomization
        training_sampler = torch.utils.data.SubsetRandomSampler(range(len(train_dataset)))

        # Prepare Data Loaders for training and validation
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                        sampler=training_sampler,
                                                        pin_memory=True, num_workers=self.n_loader_workers)

        # Set the data size:
        if 'OverlappingPatches' in config['post-load-transforms']:
            resize_config = config['post-load-transforms']['OverlappingPatches']
            if resize_config['vectorize']:
                self.data_dim = resize_config['patch-size'] ** 2
            else:
                self.data_dim = (resize_config['patch-size'], resize_config['patch-size'])
        elif 'image-size' in config:
            self.data_dim = (config['image-size'], config['image-size'])

    def parse_config(self, config):
        if 'data-dir' in config:
            self.data_dir = config['data-dir']
        else:
            raise ValueError('"data-dir" not found, but is a required key for "data-config".')

        # if 'name' in config:
        #     self.dataset_name = config['name']
        # else:
        #     raise ValueError('"name" not found, but is a required key for "data-config".')

        if 'allow-download' in config:
            self.allow_download = config['allow-download']
        else:
            self.allow_download = False

        if 'normalize-value' in config:
            self.norm_val = (config['normalize-value'], config['normalize-value'])
        else:
            self.norm_val = self._default_normalize_value

        if 'target-list' in config:
            self.target_list = config['target-list']
        else:
            self.target_list = self._default_target_list

        if 'batch-size' in config:
            self.batch_size = config['batch-size']
        else:
            self.batch_size = self._default_batch_size

        if 'n-loader-workers' in config:
            self.n_loader_workers = config['n-loader-workers']
        else:
            self.n_loader_workers = self._default_num_workers
