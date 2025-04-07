import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.distributions import Normal, Independent, kl
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl, MultivariateNormal
from torchmetrics.functional.classification import multiclass_calibration_error
from probabilistic_unet.model.prior import *
from probabilistic_unet.model.posterior import *
import sys
sys.path.insert(2, '../dataLoaders')
from probabilistic_unet.dataloader.mapillary_intended_objs import *
from sklearn.preprocessing import normalize


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(m.bias, std=0.001)

        
class ProUNet(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, num_classes, gecoConfig, device, LatentVarSize = 6, beta = 5., training = True, num_samples = 16):
        super(ProUNet, self).__init__()
        #Vars init
        self.LatentVarSize = LatentVarSize
        self.beta = beta
        self.training = training
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.gecoConfig = gecoConfig
        self.device = device
        
        if self.gecoConfig["enable"]:
            # self.geco = GECO(goal = gecoConfig["goal"], alpha = gecoConfig["alpha"], speedup = gecoConfig["speedup"], beta_init = gecoConfig["beta_init"], step_size = gecoConfig["step_size"], device = self.device)
            self.geco = MyGECO(                
                goal_fri = gecoConfig["goal_fri"],
                goal_seg = gecoConfig["goal_seg"],
                alpha = gecoConfig["alpha"],
                beta = gecoConfig["beta"],
                step_size = gecoConfig["step_size"],
                lambda_s = gecoConfig["lambda_s"],
                lambda_f = gecoConfig["lambda_f"],
                speedup = gecoConfig["speedup"],
                beta_min = gecoConfig["beta_min"],
                beta_max = gecoConfig["beta_max"])


        #architecture
        self.prior = prior(self.num_samples, self.num_classes, self.LatentVarSize).apply(init_weights)
        if training:
            self.posterior = posterior(self.num_samples, self.num_classes, self.LatentVarSize).apply(init_weights)
        
        #loss functions
        self.criterion = CrossEntopy(label_smoothing = 0.4)
        self.regressionLoss = MyMSE()

        
    def forward(self, inputImg, segmasks = None, friLabel = None):
        
        posteriorDists = self.posterior(torch.cat((inputImg, segmasks, friLabel), 1))
        seg, priorDists, fri = self.prior(inputImg, postDist = posteriorDists)
    
        return seg, priorDists, posteriorDists, fri
        
        
    def inference(self, inputFeatures):
        with torch.no_grad():
            return self.prior.inference(inputFeatures)
    
    
    def latentVisualize(self, inputFeatures, sampleLatent1 = None, sampleLatent2 = None, sampleLatent3 = None):
        return self.prior.latentVisualize(inputFeatures, sampleLatent1, sampleLatent2, sampleLatent3)

    def evaluation(self, inputFeatures, segmasks, friLabel):
        
        with torch.no_grad():
            samples, priors, fris = self.prior.inference(inputFeatures)
            posteriorDists = self.posterior.inference(torch.cat((inputFeatures, segmasks, friLabel), 1))
            return samples, priors, posteriorDists, fris
    
    
    def rec_loss(self, img, seg):

        error = self.criterion(output = img, target = seg)
        return error

    
    def kl_loss(self, priors, posteriors):
        
        klLoss = {}
        for level, (posterior, prior) in enumerate(zip(posteriors.items(), priors.items())):
            klLoss[level] = torch.mean(kl.kl_divergence(posterior[1], prior[1]), (1,2))
        return klLoss
    
    
    def elbo_loss(self, label, seg, priors, posteriors, friLabel = None, friPred = None):
        
        rec_loss = torch.mean(self.rec_loss(label, seg))
        
        kl_losses = self.kl_loss(priors, posteriors)
        kl_mean = torch.mean(torch.sum(torch.stack([i for i in kl_losses.values()]), 0))
        
        regLoss = self.regressionLoss(target = friLabel, output = friPred)
        
        loss = torch.mean(rec_loss + (self.beta * kl_mean) + regLoss)
#         loss = torch.mean(rec_loss + self.beta * kl_mean)
        
        return loss, kl_mean, kl_losses, rec_loss, regLoss

    def stats(self, predictions, labels):
        
        
        miou, ious = multiclass_iou(predictions, labels, classIds)
        CM = BatchImageConfusionMatrix(predictions, labels, classIds)
        
    
        l1Loss = multiclass_calibration_error(preds = predictions, target = torch.argmax(labels, 1), num_classes = self.num_classes, n_bins = 10, norm = 'l1')
        l2Loss = multiclass_calibration_error(preds = predictions, target = torch.argmax(labels, 1), num_classes = self.num_classes, n_bins = 10, norm = 'l2')
        l3Loss = multiclass_calibration_error(preds = predictions, target = torch.argmax(labels, 1), num_classes = self.num_classes, n_bins = 10, norm = 'max')
    
        return miou, ious, l1Loss, l2Loss, l3Loss, CM
    
    def lossGECO(self, label, segPred, priors, posteriors, friLabel = None, friPred = None):
        
        rec_loss = torch.mean(self.rec_loss(label, segPred))
        
        kl_losses = self.kl_loss(priors, posteriors)
        kl_mean = torch.mean(torch.sum(torch.stack([i for i in kl_losses.values()]), 0))
        
        regLoss = self.regressionLoss(target = friLabel, output = friPred)
 
        loss = self.geco.loss(loss_fri = regLoss, loss_seg = rec_loss, kl_loss = kl_mean)
    
        return loss, kl_mean, kl_losses, rec_loss, regLoss
    

    def loss(self, label, segPred, priors, posteriors, friLabel = None, friPred = None):
        
        
        if self.gecoConfig["enable"]:
            loss, kl_mean, kl_losses, rec_loss, regLoss = self.lossGECO(label, segPred, priors, posteriors, friLabel, friPred)
        else:
            loss, kl_mean, kl_losses, rec_loss, regLoss = self.elbo_loss(label, segPred, priors, posteriors, friLabel, friPred)

            
        miou, ious, l1Loss, l2Loss, l3Loss, CM = self.stats(segPred, label)
        
        return loss, kl_mean, kl_losses, rec_loss, miou, ious, l1Loss, l2Loss, l3Loss, regLoss, CM


    
        
        
