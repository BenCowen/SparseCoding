import torch as th
import torch.nn as nn


class InvertibleBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(InvertibleBatchNorm1d, self).__init__()
        self.batchnorm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.batchnorm(x)

    def inverse(self, x):
        # Invert the normalization
        mean = self.batchnorm.running_mean
        var = self.batchnorm.running_var
        if self.batchnorm.affine:
            weight = self.batchnorm.weight
            bias = self.batchnorm.bias
        else:
            weight = th.ones_like(var)
            bias = th.zeros_like(mean)

        # Reapply mean/variance to revert normalization
        x = x * th.sqrt(var + self.batchnorm.eps) / weight
        x = x + mean - bias / weight
        return x

