import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal, Independent, kl
import torchvision
import torchvision.transforms as T
from torch.distributions import Normal, Independent, kl, MultivariateNormal
from ResidualBlock import *

class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, padding = 1, latent_dim = None, ResLayers = 2):
        super(DownConvBlock, self).__init__()
        Reslayers = []
        self.latent_dim = latent_dim
        
        self.firstLayer = nn.Conv2d(input_dim, output_dim, kernel_size = 3, stride = 2, padding = int(padding), dilation = 1)
        self.relu = nn.ReLU()
        for _ in range(ResLayers):
            Reslayers.append(ResidualBlock(output_dim))
        
        self.layers = nn.Sequential(*Reslayers)
        
        if self.latent_dim:
            self.distLayer = nn.Conv2d(output_dim, 2 * self.latent_dim, (1,1), stride=1, dilation = 1)
    
    def forward(self, inputFeatures):
        
        emb = self.relu(self.firstLayer(inputFeatures))
        out = self.layers(emb)
        
        if self.latent_dim:
            
            mu_log_sigma = self.distLayer(out)
            mu_log_sigma = torch.squeeze(torch.squeeze(mu_log_sigma, dim=2), dim=2)
            mu = mu_log_sigma[:,:self.latent_dim]
            log_sigma = mu_log_sigma[:,self.latent_dim:]
            dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
            
            return out, dist
        
        return out    
        
        
class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, padding = 1, latent_dim = None, ResLayers = 2):
        super(UpConvBlock, self).__init__()
        Reslayers = []
        self.latent_dim = latent_dim
        
        self.firstLayer = nn.Sequential(nn.ConvTranspose2d(input_dim, output_dim, kernel_size = 4, stride = 2, padding = int(padding)),nn.ReLU(inplace=True))
        
        for _ in range(ResLayers):
            Reslayers.append(ResidualBlock(output_dim))
        
        self.layers = nn.Sequential(*Reslayers)
        
        if self.latent_dim:
            self.distLayer = nn.Sequential(nn.Conv2d(output_dim, 2 * self.latent_dim, (1,1), stride=1),
                                           nn.ReLU(inplace=True))

    def forward(self, inputFeatures):
            
        emb = self.firstLayer(inputFeatures)
        out = self.layers(emb)
        
        if self.latent_dim:
            
            mu_log_sigma = self.distLayer(out)
            mu_log_sigma = torch.squeeze(torch.squeeze(mu_log_sigma, dim=2), dim=2)
            mu = mu_log_sigma[:,:self.latent_dim].clone()
            log_sigma = mu_log_sigma[:,self.latent_dim:].clone()
            try:
                guass = Normal(loc=mu, scale=torch.exp(log_sigma))
                dist = Independent(guass,1)
            except Exception as e:
                
                print(str(e))
                ok = Normal.arg_constraints["scale"].check(torch.exp(log_sigma))
                bad_elements = torch.exp(log_sigma)[~ok]
                print("\n\nThese are the invalid values:\n{}".format(bad_elements))
            
            return out, dist
            
        return out