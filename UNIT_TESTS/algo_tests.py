"""
Checks that FISTA converges and gets the right answer on a toy problem

@author: Benjamin Cowen
@date: Feb 8 2023
@contact: benjamin.cowen.math@gmail.com
"""
import os
import torch
from math import sqrt
import UTILS.project_control as proj_control


def encoder_test(encoder_class, encoder_args,
                 make_plots=False, verbose=False, image_dir=None):
    if make_plots:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('qtagg')

    proj_control.reproducibility_mode()

    # Set up encoder object
    encoder = encoder_class(**encoder_args)
    # fista = FISTA(data_len, code_len, n_iters, sparsity_weight)

    # Set up toy data
    num_nonzero = 15
    batch_size = 23
    sig_amp = 2
    noise_var = 1
    x_opt = torch.zeros(batch_size, encoder_args['code_len'])
    nonzeros = torch.randint(0, encoder_args['code_len'], (num_nonzero,))
    x_opt[:, nonzeros] = sig_amp * (1 + torch.randn(batch_size, num_nonzero))

    with torch.no_grad():
        y0 = encoder.WeT(x_opt) + sqrt(noise_var) * torch.randn(batch_size, encoder_args['data_len'])

    # Encode
    x_est = encoder(y0)

    # Test convergence
    n_avg_convergence = 10
    loss_change = (encoder.lossHist[-1 * n_avg_convergence:] -
                   encoder.lossHist[-1 * (n_avg_convergence + 1):-1]
                   ).abs().mean().item()

    # Test reconstruction error
    rel_recon_err = ((x_est - x_opt).pow(2).sum() / x_opt.pow(2).sum()).detach().item()

    if make_plots:
        fig, ax = plt.subplots(3, 1, figsize=(5, 10))
        ax[0].stem(x_opt[0].detach(), markerfmt='bo', label='opt')
        ax[0].stem(x_est[0].detach(), markerfmt='r+', label='est')
        ax[0].legend()
        ax[0].set_title('FISTA-est vs opt')

        ax[1].stem(y0[0].detach(), markerfmt='bo', label='original')
        ax[1].stem(encoder.WeT(x_est)[0].detach(), markerfmt='r+', label='noisy')
        ax[1].legend()
        ax[1].set_title('FISTA-est vs opt')

        ax[2].semilogy(encoder.lossHist.numpy())
        ax[2].set_title('loss fcn hist')
        ax[2].set_xlabel('FISTA iterations')
        ax[2].grid()

        fig.tight_layout()
        plt.savefig(os.path.join(image_dir, 'fista-test-visualization.png'))

    if verbose:
        print('loss change = ', loss_change)
        print('reconError = ', rel_recon_err)

    return loss_change, rel_recon_err
