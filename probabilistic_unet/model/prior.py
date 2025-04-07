import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal, Independent, kl
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl, MultivariateNormal
from probabilistic_unet.model.unet_blocks import *
from crfseg import CRF



class TempSoftmax(nn.Module):
    def __init__(self, temperature, dim = 1):
        super(TempSoftmax, self).__init__()
        
        self.temperature = temperature
        self.softmax = nn.Softmax(dim = dim)
        
    def forward(self, inp):
        
        scaled_logits = inp / self.temperature
        softmax_output = self.softmax(scaled_logits)

        return softmax_output


class prior(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, num_samples, num_classes, LatentVarSize = 6):
        super(prior, self).__init__()
        #Vars init
        self.LatentVarSize = LatentVarSize
        self.num_samples = num_samples
        self.num_classes = num_classes

        
        #architecture
        self.DownConvBlock1 = DownConvBlock(input_dim = 3, output_dim = 128, ResLayers = 2, padding = 1)
        self.DownConvBlock2 = DownConvBlock(input_dim = 128, output_dim = 256, ResLayers = 2, padding = 1)
        self.DownConvBlock3 = DownConvBlock(input_dim = 256, output_dim = 512, ResLayers = 2, padding = 1)
        self.DownConvBlock4 = DownConvBlock(input_dim = 512, output_dim = 256, ResLayers = 2, latent_dim = LatentVarSize, padding = 1)

        self.UpConvBlock1 = UpConvBlock(input_dim = 256, output_dim = 512, ResLayers = 2, padding = 1)
        self.UpConvBlock2 = UpConvBlock(input_dim = (512 * 2) + LatentVarSize, output_dim = 256, ResLayers = 2, latent_dim = LatentVarSize, padding = 1)
        self.UpConvBlock3 = UpConvBlock(input_dim = (256 * 2) + LatentVarSize, output_dim = 128, ResLayers = 2, latent_dim = LatentVarSize, padding = 1)
        self.UpConvBlock4 = UpConvBlock(input_dim = (128 * 2) + LatentVarSize, output_dim = self.num_classes, ResLayers = 2, padding = 1)
        
        self.regressionLayer = UpConvBlock(input_dim = (128 * 2) + LatentVarSize, output_dim = 1, ResLayers = 2, padding = 1)
        self.softmax = nn.Softmax(dim=1)
        self.crf = CRF(n_spatial_dims=2, returns = "logits", requires_grad = True)
        

        
    def forward(self, inputFeatures, postDist):
        
        dists = {}
        encoderOuts = {}
        
        encoderOuts["out1"] = self.DownConvBlock1(inputFeatures)
        encoderOuts["out2"] = F.dropout2d(self.DownConvBlock2(encoderOuts["out1"]), p = 0.5, training = True, inplace = False)
        encoderOuts["out3"] = F.dropout2d(self.DownConvBlock3(encoderOuts["out2"]), p = 0.3, training = True, inplace = False)
        encoderOuts["out4"], dists["dist1"] = self.DownConvBlock4(encoderOuts["out3"])
        

        out = self.UpConvBlock1(encoderOuts["out4"])
        latent1 = torch.nn.Upsample(size=encoderOuts["out3"].shape[2:], mode='nearest')(postDist["dist1"].rsample())
        out = torch.cat((encoderOuts["out3"], out, latent1), 1)

        out = F.dropout2d(out, p = 0.5, training = True, inplace = False)
        out, dists["dist2"] = self.UpConvBlock2(out)
        latent2 = torch.nn.Upsample(size=encoderOuts["out2"].shape[2:], mode='nearest')(postDist["dist2"].rsample())
        out = torch.cat((encoderOuts["out2"], out, latent2), 1)

        out = F.dropout2d(out, p = 0.5, training = True, inplace = False)
        out, dists["dist3"] = self.UpConvBlock3(out)
        latent3 =  torch.nn.Upsample(size=encoderOuts["out1"].shape[2:], mode='nearest')(postDist["dist3"].rsample())
        out = torch.cat((encoderOuts["out1"], out, latent3), 1)
            
        
        segs = self.UpConvBlock4(out)
        segs = self.crf(segs)
        segs = self.softmax(segs)
        fric = self.regressionLayer(out)
        
        return segs, dists, fric
    
    
    
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

            for _ in range(self.num_samples):

                out = self.UpConvBlock1(encoderOuts["out4"])
                latent1 = torch.nn.Upsample(size=encoderOuts["out3"].shape[2:], mode='nearest')(dists["dist1"].sample())
                out = torch.cat((encoderOuts["out3"], out, latent1), 1)
                
                if "dist2" not in dists.keys():
                    out, dists["dist2"] = self.UpConvBlock2(out)
                else:
                    out, _ = self.UpConvBlock2(out)
                latent2 = torch.nn.Upsample(size=encoderOuts["out2"].shape[2:], mode='nearest')(dists["dist2"].sample())
                out = torch.cat((encoderOuts["out2"], out, latent2), 1)

                
                if "dist3" not in dists.keys():
                    out, dists["dist3"] = self.UpConvBlock3(out)
                else:
                    out, _ = self.UpConvBlock3(out)
                latent3 =  torch.nn.Upsample(size=encoderOuts["out1"].shape[2:], mode='nearest')(dists["dist3"].sample())
                out = torch.cat((encoderOuts["out1"], out, latent3), 1)

                segs = self.UpConvBlock4(out)
                segs = self.crf(segs)
                segs = self.softmax(segs)
                fric = self.regressionLayer(out)

                samples.append(segs)
                samplesFri.append(fric)
        
        
        return torch.stack(samples), dists, torch.stack(samplesFri)



    def latentVisualize(self, inputFeatures, sampleLatent1 = None, sampleLatent2 = None, sampleLatent3 = None):
        
        dists = {}
        encoderOuts = {}
        samples = []
        samplesFri = []
        
        encoderOuts["out1"] = self.DownConvBlock1(inputFeatures)
        encoderOuts["out2"] = F.dropout2d(self.DownConvBlock2(encoderOuts["out1"]), p = 0.5, training = True, inplace = False)
        encoderOuts["out3"] = F.dropout2d(self.DownConvBlock3(encoderOuts["out2"]), p = 0.3, training = True, inplace = False)
        encoderOuts["out4"], dists["dist1"] = self.DownConvBlock4(encoderOuts["out3"])
        
        for _ in range(self.num_samples):
            out = self.UpConvBlock1(encoderOuts["out4"])
            latent1 = torch.nn.Upsample(size=encoderOuts["out3"].shape[2:], mode='nearest')(dists["dist1"].rsample() if sampleLatent1 is None else sampleLatent1)
            out = torch.cat((encoderOuts["out3"], out, latent1), 1)

            out = F.dropout2d(out, p = 0.5, training = True, inplace = False)
            out, dists["dist2"] = self.UpConvBlock2(out)
            latent2 = torch.nn.Upsample(size=encoderOuts["out2"].shape[2:], mode='nearest')(dists["dist2"].rsample() if sampleLatent2 is None else sampleLatent2)
            out = torch.cat((encoderOuts["out2"], out, latent2), 1)

            out = F.dropout2d(out, p = 0.5, training = True, inplace = False)
            out, dists["dist3"] = self.UpConvBlock3(out)
            latent3 =  torch.nn.Upsample(size=encoderOuts["out1"].shape[2:], mode='nearest')(dists["dist3"].rsample() if sampleLatent3 is None else sampleLatent3)
            out = torch.cat((encoderOuts["out1"], out, latent3), 1)
                
            
            segs = self.UpConvBlock4(out)
            segs = self.crf(segs)
            segs = self.softmax(segs)
            fric = self.regressionLayer(out)


            samples.append(segs)
            samplesFri.append(fric)
        
        return torch.stack(samples), dists, torch.stack(samplesFri)