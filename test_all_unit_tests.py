"""
Runs all of the unit tests in UNIT_TESTS dir.

IMPORTANT NOTES:
 * Github test will fail if make_plots is True.
 * Github can't run GPU so the error tolerances should be set
    based on their value (not running on PC)

@author: Benjamin Cowen
@date: Feb 8 2023
@contact: benjamin.cowen.math@gmail.com
"""

from model_blocks.FISTA import FISTA
from UNIT_TESTS.algo_tests import encoder_test
from unittest import TestCase
import os

test_settings = {'make_plots': False,
                 'verbose': True,
                 'image_dir': os.path.join('UNIT_TESTS', 'unit_test_images')}


class EncoderTests(TestCase):
    def test_fista(self):
        fista_params = {'data_len': 64,
                        'code_len': 128,
                        'n_iters': 250,
                        'sparsity_weight': 5}
        convergence_tol = 3e-5
        rel_error_tol = 5e-2
        loss_chg, recon_err = encoder_test(FISTA,
                                           fista_params,
                                           test_settings['make_plots'],
                                           test_settings['verbose'],
                                           test_settings['image_dir'])
        self.assertTrue(loss_chg < convergence_tol)
        self.assertTrue(recon_err < rel_error_tol)


if __name__ == "__main__":
    tester = EncoderTests
    for attr_name in dir(tester):
        if attr_name.startswith('test_'):
            getattr(tester, attr_name)()
