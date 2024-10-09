"""
Config reader and wrapper for the whole system.

@author Benjamin Cowen
@date 3 April 2023
@contact benjamin.cowen.math@gmail.com
"""

import os
import yaml
import argparse
from lib.core.backbone import BackBone
import lib.UTILS.rng_control as rng_control

parser = argparse.ArgumentParser(description="Sparse Coding Library")

parser.add_argument('--config-file', metavar="c", type=str,
                    default=os.path.join('experiments', 'train-linear-dict.yml'))

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_file, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Add config file path to config:
    config['config-path'] = args.config_file

    # Wrangle RNG's
    rng_control.set_resproducibility(config)

    # Parse and run the config.
    controller = BackBone(config)

    # Check whether we need to continue from where we left off:
    controller.check_for_continuation()
    controller.setup_results_dir()

    # Execute the experiment
    controller.execute()
