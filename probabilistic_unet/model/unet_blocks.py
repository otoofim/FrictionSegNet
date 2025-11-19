import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from probabilistic_unet.model.residual_block import ResidualBlock

class DownConvBlock(nn.Module):
    """
    A dynamic block of convolutional layers where each layer is followed by a non-linear activation function.
    The number of layers and other parameters are dynamically configurable.
    """
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=2, padding=1, latent_dim=None, num_res_layers=2, activation=nn.ReLU):
        super(DownConvBlock, self).__init__()
        self.latent_dim = latent_dim

        self.firstLayer = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = activation()

        self.layers = nn.Sequential(
            *[ResidualBlock(output_dim, activation=activation) for _ in range(num_res_layers)]
        )

        if self.latent_dim:
            self.distLayer = nn.Conv2d(output_dim, 2 * self.latent_dim, kernel_size=1, stride=1)

    def forward(self, inputFeatures):
        emb = self.activation(self.firstLayer(inputFeatures))
        out = self.layers(emb)

        if self.latent_dim:
            mu_log_sigma = self.distLayer(out)
            mu_log_sigma = torch.squeeze(torch.squeeze(mu_log_sigma, dim=2), dim=2)
            mu = mu_log_sigma[:, :self.latent_dim]
            log_sigma = mu_log_sigma[:, self.latent_dim:]
            dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)

            return out, dist

        return out

class UpConvBlock(nn.Module):
    """
    A dynamic block consisting of an upsampling layer followed by a convolutional layer and residual layers.
    The number of layers and other parameters are dynamically configurable.
    """
    def __init__(self, input_dim, output_dim, kernel_size=4, stride=2, padding=1, latent_dim=None, num_res_layers=2, activation=nn.ReLU):
        super(UpConvBlock, self).__init__()
        self.latent_dim = latent_dim

        self.firstLayer = nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            activation(inplace=True)
        )

        self.layers = nn.Sequential(
            *[ResidualBlock(output_dim, activation=activation) for _ in range(num_res_layers)]
        )

        if self.latent_dim:
            self.distLayer = nn.Sequential(
                nn.Conv2d(output_dim, 2 * self.latent_dim, kernel_size=1, stride=1),
                activation(inplace=True)
            )

    def forward(self, inputFeatures):
        emb = self.firstLayer(inputFeatures)
        out = self.layers(emb)

        if self.latent_dim:
            mu_log_sigma = self.distLayer(out)
            mu_log_sigma = torch.squeeze(torch.squeeze(mu_log_sigma, dim=2), dim=2)
            mu = mu_log_sigma[:, :self.latent_dim]
            log_sigma = mu_log_sigma[:, self.latent_dim:]
            dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)

            return out, dist

        return out