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


def make_sparse_data(n_samples, data_len, code_len, percent_nonzero, noise_level, signal_level, encoder):
    num_nonzero = round(percent_nonzero * code_len)
    code_opt = torch.zeros(n_samples, code_len)
    nonzeros = torch.randint(0, code_len, (num_nonzero,))
    code_opt[:, nonzeros] = signal_level * (1 + torch.randn(n_samples, num_nonzero))

    with torch.no_grad():
        train_data = encoder.WeT(code_opt) + sqrt(noise_level) * torch.randn(n_samples, data_len)
    return code_opt, train_data


def encoder_loss_mean(loss_hist, n_avg_convergence):
    return (loss_hist[-1 * n_avg_convergence:] -
            loss_hist[-1 * (n_avg_convergence + 1):-1]).abs().mean().item()


def recon_rel_error(opt_codes, est_codes):
    return ((est_codes - opt_codes).pow(2).sum() / opt_codes.pow(2).sum()).detach().item()


def encoder_test(encoder_class, encoder_args,
                 make_plots=False, verbose=False, image_dir=None):
    if make_plots:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('qtagg')
    encoder_name = encoder_class.__name__
    proj_control.reproducibility_mode()

    # Lots of iters for convergence here; less for learnable later:
    encoder_args['n_iters'] = 250

    # Set up encoder object
    encoder = encoder_class(**encoder_args, trainable=False)

    # Set up training data
    percent_nonzero = 0.12
    noise_level = 1
    signal_level = 2
    batch_size = 23
    opt_train_codes, train_data = make_sparse_data(batch_size, encoder_args['data_len'], encoder_args['code_len'],
                                                   percent_nonzero, noise_level, signal_level, encoder)

    # Encode
    fixed_code_est = encoder(train_data)

    # Now train a learnable encoder to predict the output of the converged, fixed encoder
    n_train_iter = 1000
    learnable_encoder = encoder.get_copy(n_iters=10, trainable=True)
    optimizer = torch.optim.Adam(learnable_encoder.parameters(), lr=0.1)
    train_loss_hist = torch.full((n_train_iter,), torch.nan)
    loss_fcn = torch.nn.MSELoss()
    for epoch in range(n_train_iter):
        optimizer.zero_grad()
        # Forward:
        learned_code_est = learnable_encoder(train_data)
        loss = loss_fcn(learned_code_est, fixed_code_est)
        # Backward:
        loss.backward()
        optimizer.step()
        train_loss_hist[epoch] = loss.item()

    # Compute final train codes:
    learned_code_est = learnable_encoder(train_data)

    # Algo convergence
    n_avg_convergence = 2
    fixed_loss_change = encoder_loss_mean(encoder.loss_hist, n_avg_convergence)
    learned_loss_change = encoder_loss_mean(learnable_encoder.loss_hist, n_avg_convergence)
    training_loss_change = encoder_loss_mean(train_loss_hist, n_avg_convergence)

    # Reconstruction error
    fixed_rel_recon_err = recon_rel_error(opt_train_codes, fixed_code_est)
    learned_rel_recon_err = recon_rel_error(opt_train_codes, learned_code_est)

    if make_plots:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        ax[0, 0].stem(opt_train_codes[0].detach(), markerfmt='ko', label='ground truth')
        ax[0, 0].stem(fixed_code_est[0].detach(), markerfmt='r+', label='{} est'.format(encoder_name))
        ax[0, 0].stem(learned_code_est[0].detach(), markerfmt='bx', label='L-{} est'.format(encoder_name))
        ax[0, 0].legend()
        ax[0, 0].set_title('{} encodings'.format(encoder_name))

        ax[0, 1].stem(train_data[0].detach(), markerfmt='ko', label='ground truth')
        ax[0, 1].stem(encoder.WeT(fixed_code_est)[0].detach(), markerfmt='r+', label='{} recon'.format(encoder_name))
        ax[0, 1].stem(encoder.WeT(learned_code_est)[0].detach(), markerfmt='bx', label='L-{} recon'.format(encoder_name))
        ax[0, 1].legend()
        ax[0, 1].set_title('{} Reconstructions'.format(encoder_name))

        ax[1, 0].semilogy(encoder.loss_hist.numpy(), 'r', label='{}'.format(encoder_name))
        ax[1, 0].semilogy(learnable_encoder.loss_hist.numpy(), 'b', label='L-{}'.format(encoder_name))
        ax[1, 0].set_title('loss function history')
        ax[1, 0].set_xlabel('{} iterations'.format(encoder_name))
        ax[1, 0].legend()
        ax[1, 0].grid()

        ax[1, 1].semilogy(train_loss_hist, 'b', )
        ax[1, 1].set_title('Learned-{} training loss history'.format(encoder_name))
        ax[1, 1].set_xlabel('{} iterations'.format(encoder_name))
        ax[1, 1].grid()

        fig.tight_layout()
        plt.savefig(os.path.join(image_dir, '{}-test-visualization.png'.format(encoder_name)))

    if verbose:
        print('{} loss change = '.format(encoder_name), fixed_loss_change)
        print('{} reconError = '.format(encoder_name), fixed_rel_recon_err)
        print('Learned-{} loss change = '.format(encoder_name), learned_loss_change)
        print('Learned-{} reconError = '.format(encoder_name), learned_rel_recon_err)
        print('Learned-{} training loss change = '.format(encoder_name), training_loss_change)
    return fixed_loss_change, fixed_rel_recon_err, learned_rel_recon_err, training_loss_change
