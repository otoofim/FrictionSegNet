import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal, Independent, kl
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl, MultivariateNormal
from UNetBlocks import *



class posterior(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, num_samples, num_classes, LatentVarSize = 6):
        super(posterior, self).__init__()
        #Vars init
        self.LatentVarSize = LatentVarSize
        self.num_samples = num_samples
        self.num_classes = num_classes

        
        #architecture
        self.DownConvBlock1 = DownConvBlock(input_dim = 3 + self.num_classes + 1, output_dim = 128, ResLayers = 2, padding = 1)
        self.DownConvBlock2 = DownConvBlock(input_dim = 128, output_dim = 256, ResLayers = 2, padding = 1)
        self.DownConvBlock3 = DownConvBlock(input_dim = 256, output_dim = 512, ResLayers = 2, padding = 1)
        self.DownConvBlock4 = DownConvBlock(input_dim = 512, output_dim = 256, ResLayers = 2, latent_dim = LatentVarSize, padding = 1)

        self.UpConvBlock1 = UpConvBlock(input_dim = 256, output_dim = 512, ResLayers = 2, padding = 1)
        self.UpConvBlock2 = UpConvBlock(input_dim = (512 * 2) + LatentVarSize, output_dim = 256, ResLayers = 2, latent_dim = LatentVarSize, padding = 1)
        self.UpConvBlock3 = UpConvBlock(input_dim = (256 * 2) + LatentVarSize, output_dim = 128, ResLayers = 2, latent_dim = LatentVarSize, padding = 1)

        

        
    def forward(self, inputFeatures, postDist = None):
        
        dists = {}
        encoderOuts = {}
        
        encoderOuts["out1"] = self.DownConvBlock1(inputFeatures)
        encoderOuts["out2"] = F.dropout2d(self.DownConvBlock2(encoderOuts["out1"]), p = 0.5, training = True, inplace = False)
        encoderOuts["out3"] = F.dropout2d(self.DownConvBlock3(encoderOuts["out2"]), p = 0.3, training = True, inplace = False)
        encoderOuts["out4"], dists["dist1"] = self.DownConvBlock4(encoderOuts["out3"])
        
     
                   
        out = self.UpConvBlock1(encoderOuts["out4"])
        latent1 = torch.nn.Upsample(size=encoderOuts["out3"].shape[2:], mode='nearest')(dists["dist1"].rsample())
        out = torch.cat((encoderOuts["out3"], out, latent1), 1)

        out = F.dropout2d(out, p = 0.5, training = True, inplace = False)
        out, dists["dist2"] = self.UpConvBlock2(out)
        latent2 = torch.nn.Upsample(size=encoderOuts["out2"].shape[2:], mode='nearest')(dists["dist2"].rsample())
        out = torch.cat((encoderOuts["out2"], out, latent2), 1)
        
        out = F.dropout2d(out, p = 0.5, training = True, inplace = False)
        _, dists["dist3"] = self.UpConvBlock3(out)


        return dists
                
    
    def inference(self, inputFeatures):
        
        with torch.no_grad():
            dists = {}
            encoderOuts = {}
            samples = []
            samplesFri = []

            encoderOuts["out1"] = self.DownConvBlock1(inputFeatures)
            encoderOuts["out2"] = self.DownConvBlock2(encoderOuts["out1"])
            encoderOuts["out3"] = self.DownConvBlock3(encoderOuts["out2"])
            encoderOuts["out4"], dists["dist1"] = self.DownConvBlock4(encoderOuts["out3"])

            out = self.UpConvBlock1(encoderOuts["out4"])
            latent1 = torch.nn.Upsample(size=encoderOuts["out3"].shape[2:], mode='nearest')(dists["dist1"].sample())
            out = torch.cat((encoderOuts["out3"], out, latent1), 1)

            out, dists["dist2"] = self.UpConvBlock2(out)
            latent2 = torch.nn.Upsample(size=encoderOuts["out2"].shape[2:], mode='nearest')(dists["dist2"].sample())
            out = torch.cat((encoderOuts["out2"], out, latent2), 1)

            _, dists["dist3"] = self.UpConvBlock3(out)
                
            return dists
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
