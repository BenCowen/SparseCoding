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

test_settings = {'make_plots': True,
                 'verbose': True,
                 'image_dir': os.path.join('UNIT_TESTS', 'unit_test_images')}


class EncoderTests(TestCase):
    def test_fista(self):
        fista_params = {'data_len': 64,
                        'code_len': 128,
                        'sparsity_weight': 5}
        convergence_tol = 2e-4
        rel_error_tol = 6e-2
        fix_loss_chg, fix_recon_err, lrn_recon_err, training_loss_change = encoder_test(
                                                                                FISTA,
                                                                                fista_params,
                                                                                test_settings['make_plots'],
                                                                                test_settings['verbose'],
                                                                                test_settings['image_dir'])
        # Check that toy problem worked:
        self.assertTrue(fix_loss_chg < convergence_tol, 'FISTA failed to converge')
        self.assertTrue(fix_recon_err < rel_error_tol, 'FISTA failed to reconstruct data')
        self.assertTrue(lrn_recon_err < rel_error_tol, 'L-FISTA failed to reconstruct data')
        self.assertTrue(training_loss_change < convergence_tol, 'L-FISTA failed to train')


if __name__ == "__main__":
    tester = EncoderTests
    for attr_name in dir(tester):
        if attr_name.startswith('test_'):
            getattr(tester, attr_name)(tester)
