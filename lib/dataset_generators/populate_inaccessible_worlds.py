"""
Populates a database named "inaccessible_worlds".

Uses functions that are not committed to the Github repo
    containing passwords, paths, etc.

@author Benjamin Cowen
@date 30 April 2023
@contact benjamin.cowen.math@gmail.com
"""
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
from sql_tools import DataBase, Table
from non_committed_functions import get_data_root

###########################
desired_nrows = 800
desired_ncols = 800
# Decide on column names:
col_names = ['unique_id',
             'origin_path',
             'background_ratio',
             'grayscale_var',
             'nrow',
             'ncol']#,
             # 'data']
col_details = ['INT PRIMARY KEY',
               'VARCHAR(100) NOT NULL',
               'FLOAT',
               'FLOAT',
               'INT',
               'INT']#,
               # ' MEDIUMBLOB']

img_table = Table('images', col_names, col_details)
database = DataBase('inaccessible_worlds')
database.add_table(img_table)
###########################
to_img = transforms.ToTensor()
resize = transforms.Resize((desired_nrows, desired_ncols), interpolation=transforms.InterpolationMode.NEAREST)

# Add data to the table:
channelwise_mean_list = []
unique_id = -1

for root, subdirs, files in os.walk(get_data_root('inaccessible worlds')):
    for filename in files:
        if filename.endswith('.pt'):
            continue
        unique_id += 1

        # Load the image
        filepath = os.path.join(root, filename)
        img = to_img(Image.open(filepath))

        # Scale it to [0, 1]
        img -= img.min()
        img /= img.max()

        # Get metadata
        channelwise_mean_list.append([img[z].mean() for z in range(img.shape[0])])
        bkgd_rat = ((img == img.sum(0).view(-1).mode().values).sum() / img.numel()).item()
        grayscale_var = ((img.numpy().argmin(0)+img.numpy().argmax(0))/2).var()

        # Do some of the preprocessing now
        # img = resize(img)

        # Insert data into the table:
        data_dict = {'unique_id': unique_id,
                     'origin_path': filepath,
                     'background_ratio': bkgd_rat,
                     'grayscale_var': grayscale_var,
                     'nrow': img.shape[1],
                     'ncol': img.shape[2],
                     # 'data': img
                     }

        database.extend_table(img_table, data_dict)

# Do final analysis:
M = torch.Tensor(channelwise_mean_list)
stats = {'mean': M.mean(0),
         'var': M.var(0),
         'n-samples': M.shape[0],
         'n-channels': M.shape[1]}
torch.save(stats, os.path.join(get_data_root('inaccessible worlds'),
                               'dataset_stats.pt'))
print('channel statistics: \n', stats)

