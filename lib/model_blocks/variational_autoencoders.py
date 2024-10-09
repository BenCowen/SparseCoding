import torch
import torch.nn as nn
from collections import OrderedDict
# from lib.trainers.custom_loss_functions import ELBO
from lib.model_blocks.AlgorithmBlock import AlgorithmBlock


class Reshaper:
    def __init__(self, n_channel, feat_size):
        self.n_channel = n_channel
        self.feat_size = feat_size

    def __call__(self, x):
        return x.view(-1, self.n_channel, self.feat_size, self.feat_size)


class VanillaVae(AlgorithmBlock):
    """Variational Autoencoder"""

    def __init__(self, config, non_blocking=True):
        super(VanillaVae, self).__init__()
        self.config = config
        self._device = config['device']
        self.non_blocking = non_blocking
        input_channels = self.config['in-channels']
        hidden_sizes = self.config['hidden-sizes']
        self.latent_dim = self.config['latent-dim']
        cnn_kwargs = self.config['cnn-kwargs']
        n_block = len(hidden_sizes)
        encoder = OrderedDict([])
        decoder = OrderedDict([])
        in_channels = input_channels
        for block_id, out_channels in enumerate(hidden_sizes):
            pre = f"e{block_id}"
            encoder.update([(pre + "conv", nn.Conv2d(
                in_channels, out_channels, **cnn_kwargs)),
                            (pre + "bnrm", nn.BatchNorm2d(out_channels)),
                            (pre + "gelu", nn.GELU())])
            in_channels = out_channels

        # Encoder needs a flatten as well:
        encoder.update({'flat': nn.Flatten()})
        self.encoder = nn.Sequential(encoder)
        # Presumes the final feature map has 4 elements:
        self.fc_mu = nn.Linear(out_channels * 4, self.latent_dim )
        self.fc_var = nn.Linear(out_channels * 4, self.latent_dim )

        self.reshape = Reshaper(out_channels, 2)
        self.decoder_proj = nn.Linear(self.latent_dim , out_channels * 4)

        for block_id, out_channels in enumerate(reversed(hidden_sizes)):
            pre = f"d{n_block - block_id}"
            decoder.update([(pre + "conv", nn.ConvTranspose2d(
                in_channels, out_channels, **cnn_kwargs)),
                            (pre + "bnrm", nn.BatchNorm2d(out_channels)),
                            (pre + "gelu", nn.GELU()),
                            ])
            in_channels = out_channels
        # Need one more decoder layer to get the shape right
        # self.decoder = OrderedDict(reversed(decoder.items()))
        decoder.update([
            ('finalTConv', nn.ConvTranspose2d(in_channels, input_channels,
                                              kernel_size=cnn_kwargs['kernel_size']-1,
                                              stride=cnn_kwargs['stride'],
                                              padding=cnn_kwargs['padding'])),
            ("finalBnrm", nn.BatchNorm2d(input_channels)),
            # ("finalGelu", nn.GELU()),
            # ("finalConv", nn.Conv2d(in_channels, 3, **cnn_kwargs)),
            ("finalNL", nn.Tanh())
        ])
        self.decoder = nn.Sequential(decoder)

        if self.config['print']:
            print(self.encoder)
            print(self.fc_mu)
            print(self.fc_var)
            print(self.decoder)

    def encode(self, x):
        """ Encoder image to latent codes """
        feats = self.encoder(x)
        mu = self.fc_mu(feats)
        log_var = self.fc_var(feats)
        return [mu, log_var]

    def decode(self, z):
        """ Decodes latent code into image"""
        return self.decoder(self.reshape(self.decoder_proj(z)))

    def sample(self, mu, log_var):
        """ Indirectly sample a point from given gaussian"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sample(mu, log_var)
        return {'recon': self.decode(z), 'mu': mu, 'log_var': log_var}


if __name__ == "__main__":
    in_shape = 128
    x = torch.ones(5, 3, in_shape, in_shape)
    config = {'in-channels': 3,
              'hidden-sizes': [16, 16, 32, 32, 128, 128],  #[32, 64, 128, 256, 512], #
              'latent-dim': 10,
              'cnn-kwargs': dict(kernel_size=3, stride=2, padding=1),
              'print': False
              }

    f = VanillaVae(config)


    # Print every intermediate shape:
    for (name, layer) in f.encoder.named_children():
        print(f'Input to {name} has shape {x.shape}')
        x = layer(x)

    muu = f.fc_mu(x)
    logvar = f.fc_var(x)
    print(f'mu has shape {muu.shape}, log_var has {logvar.shape}')

    z = f.sample(muu, logvar)
    print(f'sampled shape {z.shape}')
    z = f.decoder_proj(z)
    print(f'dec proj shape {z.shape}')
    z = f.reshape(z)
    for (name, layer) in f.decoder.named_children():
        print(f'Input to {name} has shape {z.shape}')
        z = layer(z)

    print(f'final shape: {z.shape}')
    ff = ELBO()
    x = torch.ones(5, 3, in_shape, in_shape)
    ff(x, *f(x))

