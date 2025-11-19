import torch
import torch.nn as nn
import torch.nn.functional as F
from probabilistic_unet.model.unet_blocks import DownConvBlock, UpConvBlock
from TorchCRF import CRF


class TempSoftmax(nn.Module):
    def __init__(self, temperature, dim=1):
        super(TempSoftmax, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, inp):
        scaled_logits = inp / self.temperature
        softmax_output = self.softmax(scaled_logits)
        return softmax_output

class Prior(nn.Module):
    """
    A dynamic prior model with configurable architecture.
    """
    def __init__(self, num_samples, num_classes, latent_var_size=6, input_dim=3, base_channels=128, num_res_layers=2, activation=nn.ReLU):
        super(Prior, self).__init__()
        self.latent_var_size = latent_var_size
        self.num_samples = num_samples
        self.num_classes = num_classes

        # Architecture
        self.down_blocks = nn.ModuleList([
            DownConvBlock(input_dim=input_dim, output_dim=base_channels, num_res_layers=num_res_layers, activation=activation),
            DownConvBlock(input_dim=base_channels, output_dim=base_channels * 2, num_res_layers=num_res_layers, activation=activation),
            DownConvBlock(input_dim=base_channels * 2, output_dim=base_channels * 4, num_res_layers=num_res_layers, activation=activation),
            DownConvBlock(input_dim=base_channels * 4, output_dim=base_channels * 2, latent_dim=latent_var_size, num_res_layers=num_res_layers, activation=activation)
        ])

        self.up_blocks = nn.ModuleList([
            UpConvBlock(input_dim=base_channels * 2, output_dim=base_channels * 4, num_res_layers=num_res_layers, activation=activation),
            UpConvBlock(input_dim=(base_channels * 4) + latent_var_size, output_dim=base_channels * 2, latent_dim=latent_var_size, num_res_layers=num_res_layers, activation=activation),
            UpConvBlock(input_dim=(base_channels * 2) + latent_var_size, output_dim=base_channels, latent_dim=latent_var_size, num_res_layers=num_res_layers, activation=activation),
            UpConvBlock(input_dim=(base_channels) + latent_var_size, output_dim=num_classes, num_res_layers=num_res_layers, activation=activation)
        ])

        self.softmax = nn.Softmax(dim=1)
        self.crf = CRF(num_classes)

    def forward(self, input_features, post_dist):
        dists = {}
        encoder_outs = {}

        # Downsampling
        for i, block in enumerate(self.down_blocks):
            if i == 0:
                encoder_outs[f"out{i+1}"] = block(input_features)
            else:
                encoder_outs[f"out{i+1}"] = F.dropout2d(block(encoder_outs[f"out{i}"]), p=0.5 if i == 1 else 0.3, training=self.training)

        encoder_outs["out4"], dists["dist1"] = self.down_blocks[-1](encoder_outs["out3"])

        # Upsampling
        out = self.up_blocks[0](encoder_outs["out4"])
        for i in range(1, len(self.up_blocks)):
            latent = torch.nn.Upsample(size=encoder_outs[f"out{3-i}"].shape[2:], mode='nearest')(post_dist[f"dist{i}"].rsample())
            out = torch.cat((encoder_outs[f"out{3-i}"], out, latent), 1)
            out = F.dropout2d(out, p=0.5, training=self.training)
            out, dists[f"dist{i+1}"] = self.up_blocks[i](out)

        segs = self.up_blocks[-1](out)
        segs = self.crf(segs)
        segs = self.softmax(segs)
        return segs, dists

    def inference(self, input_features):
        with torch.no_grad():
            dists = {}
            encoder_outs = {}
            samples = []
            samples_fric = []

            # Downsampling
            for i, block in enumerate(self.down_blocks):
                if i == 0:
                    encoder_outs[f"out{i+1}"] = block(input_features)
                else:
                    encoder_outs[f"out{i+1}"] = block(encoder_outs[f"out{i}"])

            encoder_outs["out4"], dists["dist1"] = self.down_blocks[-1](encoder_outs["out3"])

            # Sampling
            for _ in range(self.num_samples):
                out = self.up_blocks[0](encoder_outs["out4"])
                for i in range(1, len(self.up_blocks)):
                    latent = torch.nn.Upsample(size=encoder_outs[f"out{3-i}"].shape[2:], mode='nearest')(dists[f"dist{i}"].sample())
                    out = torch.cat((encoder_outs[f"out{3-i}"], out, latent), 1)
                    if f"dist{i+1}" not in dists:
                        out, dists[f"dist{i+1}"] = self.up_blocks[i](out)
                    else:
                        out, _ = self.up_blocks[i](out)

                segs = self.up_blocks[-1](out)
                segs = self.crf(segs)
                segs = self.softmax(segs)

                samples.append(segs)
                
        return torch.stack(samples), dists, torch.stack(samples_fric)

    def latentVisualize(self, input_features, sample_latent1=None, sample_latent2=None, sample_latent3=None):
        """
        Visualize the latent space by sampling or using provided latent variables.
        """
        dists = {}
        encoder_outs = {}
        samples = []

        # Downsampling
        encoder_outs["out1"] = self.down_blocks[0](input_features)
        encoder_outs["out2"] = F.dropout2d(self.down_blocks[1](encoder_outs["out1"]), p=0.5, training=self.training)
        encoder_outs["out3"] = F.dropout2d(self.down_blocks[2](encoder_outs["out2"]), p=0.3, training=self.training)
        encoder_outs["out4"], dists["dist1"] = self.down_blocks[3](encoder_outs["out3"])

        # Sampling and visualization
        for _ in range(self.num_samples):
            out = self.up_blocks[0](encoder_outs["out4"])
            latent1 = torch.nn.Upsample(size=encoder_outs["out3"].shape[2:], mode='nearest')(
                dists["dist1"].rsample() if sample_latent1 is None else sample_latent1
            )
            out = torch.cat((encoder_outs["out3"], out, latent1), 1)

            out = F.dropout2d(out, p=0.5, training=self.training)
            out, dists["dist2"] = self.up_blocks[1](out)
            latent2 = torch.nn.Upsample(size=encoder_outs["out2"].shape[2:], mode='nearest')(
                dists["dist2"].rsample() if sample_latent2 is None else sample_latent2
            )
            out = torch.cat((encoder_outs["out2"], out, latent2), 1)

            out = F.dropout2d(out, p=0.5, training=self.training)
            out, dists["dist3"] = self.up_blocks[2](out)
            latent3 = torch.nn.Upsample(size=encoder_outs["out1"].shape[2:], mode='nearest')(
                dists["dist3"].rsample() if sample_latent3 is None else sample_latent3
            )
            out = torch.cat((encoder_outs["out1"], out, latent3), 1)

            segs = self.up_blocks[3](out)
            segs = self.crf(segs)
            segs = self.softmax(segs)

            samples.append(segs)

        return torch.stack(samples), dists