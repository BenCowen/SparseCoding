"""
Runs all of the unit tests in UNIT_TESTS dir.

IMPORTANT NOTES:
 * Github test will fail if make_plots is True.
 * Github can't run GPU so the error tolerances should be set
    based on their value (not running on PC). This also means
    I had to spoof the requirements.txt file.

@author: Benjamin Cowen
@date: Feb 8 2023
@contact: benjamin.cowen.math@gmail.com
"""
from lib.model_blocks.ISTA import FISTA, ISTA
from UNIT_TESTS.unrolled_algo_tests import encoder_test
import unittest
import os


class UnrolledAlgorithmTester(unittest.TestCase):
    """
    For each unrolled algorithm, tests
        - convergence and success of fixed version
        - class methods converting fixed to learnable instance
        - convergence on training the learnable instance
    """

    def __init__(self, make_plots=True, verbose=True):
        unittest.TestCase.__init__(self)
        # Same data, same cost function:
        self.problem_params = {'data_dim': 64,
                               'code_dim': 128,
                               'sparsity_weight': 0.5,
                               'n_iter': 250}

        self.tester_settings = {'make_plots': make_plots,
                                'verbose': verbose,
                                'image_dir': os.path.join('UNIT_TESTS', 'unit_test_images'),
                                'gd_n_iters': 600}

        self._recon_rel_err_tol = 0.07
        self._convergence_rel_chg_tol = 0.001

        self.convex_solutions = {}
        # Note: in GITHUB test environment, plotting will make the tests fail.
        #       OVERRIDE this arg so I don't have to remember to change it.
        if os.getenv("GITHUB_ACTIONS") is not None:
            if "true" in os.getenv("GITHUB_ACTIONS"):
                self.tester_settings['make_plots'] = False

    def test_fista(self):
        self.tester_settings['gd_learnrate'] = 0.01
        fix_loss_chg, fix_recon_err, lrn_recon_err, training_loss_change, est_code = encoder_test(
            FISTA,
            self.problem_params,
            self.tester_settings
        )
        # Check that toy problem worked:
        self.assertTrue(fix_loss_chg < self._convergence_rel_chg_tol, 'FISTA failed to converge')
        self.assertTrue(fix_recon_err < self._recon_rel_err_tol, 'FISTA failed to reconstruct data')
        self.assertTrue(lrn_recon_err < self._recon_rel_err_tol, 'L-FISTA failed to reconstruct data')
        self.assertTrue(training_loss_change < self._convergence_rel_chg_tol, 'L-FISTA failed to train')
        # Save for later:
        self.convex_solutions['fista'] = est_code

    def test_ista(self, conv=False):
        self.tester_settings['gd_learnrate'] = 0.0025
        fix_loss_chg, fix_recon_err, lrn_recon_err, training_loss_change, est_code = encoder_test(
            ISTA,
            self.problem_params,
            self.tester_settings
        )
        # Check that toy problem worked:
        self.assertTrue(fix_loss_chg < self._convergence_rel_chg_tol, 'ISTA failed to converge')
        self.assertTrue(fix_recon_err < self._recon_rel_err_tol, 'ISTA failed to reconstruct data')
        self.assertTrue(lrn_recon_err < self._recon_rel_err_tol, 'L-ISTA failed to reconstruct data')
        self.assertTrue(training_loss_change < self._convergence_rel_chg_tol, 'L-ISTA failed to train')
        # Save for later:
        self.convex_solutions['ista'] = est_code

    def composite_test_convexity(self):
        # The different methods should be getting the same solution
        #  since they are optimizing the same convex cost function...
        algo_names = list(self.convex_solutions.keys())
        for idx1, key1 in enumerate(algo_names):
            for key2 in algo_names[(idx1 + 1):]:
                rel_recon_err = ((self.convex_solutions[key1] - self.convex_solutions[key2]).pow(2).sum().item() /
                                 (self.convex_solutions[key2].pow(2).sum().item() *
                                  self.convex_solutions[key1].shape[0]))
                print(f'\t||{key1} - {key2}||^2 / (bsz*||{key2}||^2) = {rel_recon_err:.2E}')
                self.assertTrue(rel_recon_err < self._recon_rel_err_tol,
                                f'{key1} and {key2} converged to different solutions!')

    def runTest(self):
        #########################################
        # Solver Tests
        # Component tests:
        self.test_ista()
        self.test_fista()

        # Composite tests:
        self.composite_test_convexity()

        #########################################
        # Pipeline integ test


if __name__ == "__main__":
    unittest.main()
