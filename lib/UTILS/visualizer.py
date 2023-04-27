import matplotlib.pyplot as plt
import os
import itertools


class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def tensor_preprocess(self, tensor, color=False):
        """
        Returns intensity min and max + converts tensor to numpy. Rearranges
        color dimension to final axis if necessary.
        """
        if color:
            tensor = tensor.transpose(2, 3).transpose(3, 4)
        # Convert to [0, 1]
        tensor -= tensor.min()
        tensor /= tensor.max()
        tensor = tensor.detach().cpu().numpy()
        return tensor, tensor.min(), tensor.max()

    def array_plot(self, tensor_grid, save_name, title_str='', color=False):
        """
        Assumes tensor_grid: (nr, nc, m, n)
        generates a plot with nr rows, nc cols, with mxn images shown in each.
        """
        epoch_save = os.path.join(self.save_dir, 'dict_hist')
        if not os.path.exists(epoch_save):
            os.makedirs(epoch_save)

        tensor_grid, cmin, cmax = self.tensor_preprocess(tensor_grid, color)
        fig, ax = plt.subplots(nrows=tensor_grid.shape[0],
                               ncols=tensor_grid.shape[1],
                               figsize=(10, 10))

        for m in range(tensor_grid.shape[0]):
            for n in range(tensor_grid.shape[1]):
                ax[m, n].imshow(tensor_grid[m][n],
                                vmin=cmin, vmax=cmax)
                ax[m, n].set_axis_off()

        plt.suptitle(title_str,
                     fontsize=18)
        fig.tight_layout()
        plt.savefig(os.path.join(epoch_save, save_name + '.png'),
                    bbox_inches='tight')
        plt.figure().clear()
        plt.close(fig)
        plt.cla()
        plt.clf()

    def plot_loss(self, trn_rec, save_name, title_str=''):

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

        epoch_lines = []
        last = 0
        for epoch in trn_rec['loss-hist']:
            epoch_lines.append(last + len(epoch) - 1)
            last += len(epoch) - 1
        all_loss = list(itertools.chain.from_iterable(trn_rec['loss-hist']))
        all_sparsity = list(itertools.chain.from_iterable(trn_rec['sparsity-hist']))
        ax.plot(all_loss, color='b', label='Loss')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_xlabel('Epochs')
        ax.set_xticks(epoch_lines)
        ax.set_xticklabels([n+1 for n in range(len(epoch_lines))])

        ax2 = ax.twinx()
        ax2.plot(all_sparsity, color='r', label='Sparsity')
        ax2.tick_params(axis='y', labelcolor='r')

        ax.vlines(epoch_lines, ax.get_ylim()[0],
                  ax.get_ylim()[1], color='black')

        plt.suptitle(title_str,
                     fontsize=18)
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name + '.png'),
                    bbox_inches='tight')
        plt.figure().clear()
        plt.close(fig)
        plt.cla()
        plt.clf()
