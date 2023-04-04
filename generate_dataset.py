"""
Initializes a Dataset class. Maybe just do this in regular `main` thing?..

@author Benjamin Cowen
@date 22 Jan 2022
@contact benjamin.cowen.math@gmail.com
"""

import os, sys
from argparse import ArgumentParser
from yaml import safe_load as yml_safeload

if __name__ == '__main__':
    parser = ArgumentParser(description="Generate Dataset From Config")
    parser.add_argument('-c', '--config-path', help='Path to config file',
                        default='basic-mnist-demo.yml')
    inputs = parser.parse_args(sys.argv[1:])
    
    # Load config
    config = open(inputs.config_path, 'r')
    config = yml_safeload(config.read())
    
    # Construct the dataset in the dataset wrapper class.
    Dataset(config)
    
    exit()