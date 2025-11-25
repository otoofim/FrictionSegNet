import torch
import torch.nn as nn
from torch.distributions import kl
from torchmetrics.functional.classification import multiclass_calibration_error

from probabilistic_unet.model.posterior import Posterior
from probabilistic_unet.model.prior import Prior
from probabilistic_unet.utils.confusion_matrix.confusion_matrix import (
    BatchImageConfusionMatrix,
    multiclass_iou,
)
from probabilistic_unet.utils.objective_functions.objective_function import CrossEntopy


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(m.bias, std=0.001)


class ProUNet(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """

    def __init__(
        self,
        num_classes,
        device,
        LatentVarSize=6,
        beta=5.0,
        training=True,
        num_samples=16,
    ):
        super(ProUNet, self).__init__()
        # Vars init
        self.LatentVarSize = LatentVarSize
        self.beta = beta
        self.training = training
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.device = device

        # architecture
        self.prior = Prior(
            num_samples=self.num_samples,
            num_classes=self.num_classes,
            latent_var_size=self.LatentVarSize,
            input_dim=3,
            base_channels=128,
            num_res_layers=2,
            activation=nn.ReLU,
        ).apply(init_weights)
        if training:
            self.posterior = Posterior(
                num_samples=self.num_samples,
                num_classes=self.num_classes,
                latent_var_size=self.LatentVarSize,
                input_dim=None,  # Will be calculated as 3 + num_classes
                base_channels=128,
                num_res_layers=2,
                activation=nn.ReLU,
            ).apply(init_weights)

        # loss functions
        self.criterion = CrossEntopy(label_smoothing=0.4)

    def forward(self, inputImg, segmasks=None):
        posteriorDists = self.posterior(torch.cat((inputImg, segmasks), 1))
        seg, priorDists = self.prior(inputImg, post_dist=posteriorDists)

        return seg, priorDists, posteriorDists

    def inference(self, inputFeatures):
        """Returns (samples, dists) where samples is stacked tensor of predictions"""
        return self.prior.inference(inputFeatures)

    def latentVisualize(
        self, inputFeatures, sampleLatent1=None, sampleLatent2=None, sampleLatent3=None
    ):
        """Returns (samples, dists) with custom latent samples if provided"""
        return self.prior.latentVisualize(
            inputFeatures,
            sample_latent1=sampleLatent1,
            sample_latent2=sampleLatent2,
            sample_latent3=sampleLatent3,
        )

    def evaluation(self, inputFeatures, segmasks):
        with torch.no_grad():
            samples, priors = self.prior.inference(inputFeatures)
            posteriorDists = self.posterior.inference(
                torch.cat((inputFeatures, segmasks), 1)
            )
            return samples, priors, posteriorDists

    def rec_loss(self, img, seg):
        error = self.criterion(output=img, target=seg)
        return error

    def kl_loss(self, priors, posteriors):
        klLoss = {}
        for level, (posterior, prior) in enumerate(
            zip(posteriors.items(), priors.items())
        ):
            klLoss[level] = torch.mean(kl.kl_divergence(posterior[1], prior[1]), (1, 2))
        return klLoss

    def elbo_loss(self, label, seg, priors, posteriors):
        rec_loss = torch.mean(self.rec_loss(label, seg))

        kl_losses = self.kl_loss(priors, posteriors)
        kl_mean = torch.mean(torch.sum(torch.stack([i for i in kl_losses.values()]), 0))

        loss = torch.mean(rec_loss + (self.beta * kl_mean))

        return loss, kl_mean, kl_losses, rec_loss

    def stats(self, predictions, labels):
        # Generate class indices dynamically based on num_classes
        class_indices = list(range(self.num_classes))

        miou, ious = multiclass_iou(predictions, labels, class_indices)
        CM = BatchImageConfusionMatrix(predictions, labels, class_indices)

        l1Loss = multiclass_calibration_error(
            preds=predictions,
            target=torch.argmax(labels, 1),
            num_classes=self.num_classes,
            n_bins=10,
            norm="l1",
        )
        l2Loss = multiclass_calibration_error(
            preds=predictions,
            target=torch.argmax(labels, 1),
            num_classes=self.num_classes,
            n_bins=10,
            norm="l2",
        )
        l3Loss = multiclass_calibration_error(
            preds=predictions,
            target=torch.argmax(labels, 1),
            num_classes=self.num_classes,
            n_bins=10,
            norm="max",
        )

        return miou, ious, l1Loss, l2Loss, l3Loss, CM

    def loss(self, label, segPred, priors, posteriors):
        loss, kl_mean, kl_losses, rec_loss = self.elbo_loss(
            label, segPred, priors, posteriors
        )

        miou, ious, l1Loss, l2Loss, l3Loss, CM = self.stats(segPred, label)

        return (
            loss,
            kl_mean,
            kl_losses,
            rec_loss,
            miou,
            ious,
            l1Loss,
            l2Loss,
            l3Loss,
            CM,
        )
