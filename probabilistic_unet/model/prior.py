import torch
import torch.nn as nn
import torch.nn.functional as F
from probabilistic_unet.model.unet_blocks import DownConvBlock, UpConvBlock
from TorchCRF import CRF


class Prior(nn.Module):
    """
    A dynamic prior model with configurable architecture.
    """

    def __init__(
        self,
        num_samples,
        num_classes,
        latent_var_size=6,
        input_dim=3,
        base_channels=128,
        num_res_layers=2,
        activation=nn.ReLU,
    ):
        super(Prior, self).__init__()
        self.latent_var_size = latent_var_size
        self.num_samples = num_samples
        self.num_classes = num_classes

        # Architecture
        self.down_blocks = nn.ModuleList(
            [
                DownConvBlock(
                    input_dim=input_dim,
                    output_dim=base_channels,
                    num_res_layers=num_res_layers,
                    activation=activation,
                ),
                DownConvBlock(
                    input_dim=base_channels,
                    output_dim=base_channels * 2,
                    num_res_layers=num_res_layers,
                    activation=activation,
                ),
                DownConvBlock(
                    input_dim=base_channels * 2,
                    output_dim=base_channels * 4,
                    num_res_layers=num_res_layers,
                    activation=activation,
                ),
                DownConvBlock(
                    input_dim=base_channels * 4,
                    output_dim=base_channels * 2,
                    latent_dim=latent_var_size,
                    num_res_layers=num_res_layers,
                    activation=activation,
                ),
            ]
        )

        self.up_blocks = nn.ModuleList(
            [
                UpConvBlock(
                    input_dim=base_channels * 2,
                    output_dim=base_channels * 4,
                    num_res_layers=num_res_layers,
                    activation=activation,
                ),
                UpConvBlock(
                    input_dim=(base_channels * 4) + latent_var_size,
                    output_dim=base_channels * 2,
                    latent_dim=latent_var_size,
                    num_res_layers=num_res_layers,
                    activation=activation,
                ),
                UpConvBlock(
                    input_dim=(base_channels * 2) + latent_var_size,
                    output_dim=base_channels,
                    latent_dim=latent_var_size,
                    num_res_layers=num_res_layers,
                    activation=activation,
                ),
                UpConvBlock(
                    input_dim=(base_channels) + latent_var_size,
                    output_dim=num_classes,
                    num_res_layers=num_res_layers,
                    activation=activation,
                ),
            ]
        )

        self.softmax = nn.Softmax(dim=1)
        self.crf = CRF(num_classes)

    def forward(self, input_features, post_dist):
        dists = {}
        encoder_outs = {}
        dist_counter = 1

        # Downsampling
        for i, block in enumerate(self.down_blocks):
            if i == 0:
                encoder_outs[f"out{i + 1}"] = block(input_features)
            else:
                encoder_outs[f"out{i + 1}"] = F.dropout2d(
                    block(encoder_outs[f"out{i}"]),
                    p=0.5 if i == 1 else 0.3,
                    training=self.training,
                )

        # Handle last down_block if it has latent_dim
        if self.down_blocks[-1].latent_dim is not None:
            (
                encoder_outs[f"out{len(self.down_blocks)}"],
                dists[f"dist{dist_counter}"],
            ) = self.down_blocks[-1](encoder_outs[f"out{len(self.down_blocks) - 1}"])
            dist_counter += 1
        else:
            encoder_outs[f"out{len(self.down_blocks)}"] = self.down_blocks[-1](
                encoder_outs[f"out{len(self.down_blocks) - 1}"]
            )

        # Upsampling - First block (no skip connection, no latent)
        result = self.up_blocks[0](encoder_outs[f"out{len(self.down_blocks)}"])
        if self.up_blocks[0].latent_dim is not None:
            out, dists[f"dist{dist_counter}"] = result
            dist_counter += 1
        else:
            out = result

        # Remaining up_blocks with skip connections
        for i in range(1, len(self.up_blocks)):
            # Get encoder skip connection (reverse order)
            skip_idx = len(self.down_blocks) - i
            encoder_skip = encoder_outs[f"out{skip_idx}"]

            # Check if this block expects latent variable
            if self.up_blocks[i].latent_dim is not None:
                # Use latent from posterior
                latent = post_dist[f"dist{i}"].rsample()

                # Find the largest spatial dimensions among the three
                shapes = [encoder_skip.shape[2:], latent.shape[2:], out.shape[2:]]
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                target_size = (max_h, max_w)

                # Upsample all three to the largest size
                encoder_skip = F.interpolate(
                    encoder_skip,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                latent = F.interpolate(latent, size=target_size, mode="nearest")
                out = F.interpolate(
                    out, size=target_size, mode="bilinear", align_corners=False
                )

                # Concatenate encoder skip, previous output, and latent
                concatenated = torch.cat((encoder_skip, out, latent), dim=1)
            else:
                # No latent expected, just concatenate encoder_skip and out
                shapes = [encoder_skip.shape[2:], out.shape[2:]]
                max_h = max(s[0] for s in shapes)
                max_w = max(s[1] for s in shapes)
                target_size = (max_h, max_w)

                encoder_skip = F.interpolate(
                    encoder_skip,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                out = F.interpolate(
                    out, size=target_size, mode="bilinear", align_corners=False
                )

                concatenated = torch.cat((encoder_skip, out), dim=1)

            # Apply dropout and pass through block
            concatenated = F.dropout2d(concatenated, p=0.5, training=self.training)
            result = self.up_blocks[i](concatenated)

            # Handle output based on whether block has latent_dim
            if self.up_blocks[i].latent_dim is not None:
                out, dists[f"dist{dist_counter}"] = result
                dist_counter += 1
            else:
                out = result

        segs = self.crf(out)
        segs = self.softmax(segs)
        return segs, dists

    def inference(self, input_features):
        with torch.no_grad():
            dists = {}
            encoder_outs = {}
            samples = []
            dist_counter = 1

            # Downsampling
            for i, block in enumerate(self.down_blocks):
                if i == 0:
                    encoder_outs[f"out{i + 1}"] = block(input_features)
                else:
                    encoder_outs[f"out{i + 1}"] = block(encoder_outs[f"out{i}"])

            # Handle last down_block if it has latent_dim
            if self.down_blocks[-1].latent_dim is not None:
                (
                    encoder_outs[f"out{len(self.down_blocks)}"],
                    dists[f"dist{dist_counter}"],
                ) = self.down_blocks[-1](
                    encoder_outs[f"out{len(self.down_blocks) - 1}"]
                )
                dist_counter += 1
            else:
                encoder_outs[f"out{len(self.down_blocks)}"] = self.down_blocks[-1](
                    encoder_outs[f"out{len(self.down_blocks) - 1}"]
                )

            # Sampling loop
            for _ in range(self.num_samples):
                # First up_block (no skip connection)
                result = self.up_blocks[0](encoder_outs[f"out{len(self.down_blocks)}"])
                if self.up_blocks[0].latent_dim is not None:
                    out, new_dist = result
                    if _ == 0:  # Only store dist on first sample
                        dists[f"dist{dist_counter}"] = new_dist
                else:
                    out = result

                # Remaining up_blocks with skip connections
                for i in range(1, len(self.up_blocks)):
                    # Get encoder skip connection (reverse order)
                    skip_idx = len(self.down_blocks) - i
                    encoder_skip = encoder_outs[f"out{skip_idx}"]

                    # Check if this block expects latent variable
                    if self.up_blocks[i].latent_dim is not None:
                        # Sample latent from the distribution
                        latent = dists[f"dist{i}"].sample()

                        # Find the largest spatial dimensions among the three
                        shapes = [
                            encoder_skip.shape[2:],
                            latent.shape[2:],
                            out.shape[2:],
                        ]
                        max_h = max(s[0] for s in shapes)
                        max_w = max(s[1] for s in shapes)
                        target_size = (max_h, max_w)

                        # Upsample all three to the largest size
                        encoder_skip = F.interpolate(
                            encoder_skip,
                            size=target_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        latent = F.interpolate(latent, size=target_size, mode="nearest")
                        out = F.interpolate(
                            out, size=target_size, mode="bilinear", align_corners=False
                        )

                        # Concatenate encoder skip, previous output, and latent
                        concatenated = torch.cat((encoder_skip, out, latent), dim=1)
                    else:
                        # No latent expected, just concatenate encoder_skip and out
                        shapes = [encoder_skip.shape[2:], out.shape[2:]]
                        max_h = max(s[0] for s in shapes)
                        max_w = max(s[1] for s in shapes)
                        target_size = (max_h, max_w)

                        encoder_skip = F.interpolate(
                            encoder_skip,
                            size=target_size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        out = F.interpolate(
                            out, size=target_size, mode="bilinear", align_corners=False
                        )

                        concatenated = torch.cat((encoder_skip, out), dim=1)

                    # Pass through block
                    result = self.up_blocks[i](concatenated)

                    # Handle output based on whether block has latent_dim
                    if self.up_blocks[i].latent_dim is not None:
                        out, new_dist = result
                        if _ == 0:  # Only store dist on first sample
                            dists[f"dist{i + 1}"] = new_dist
                    else:
                        out = result

                segs = self.crf(out)
                segs = self.softmax(segs)
                samples.append(segs)

        return torch.stack(samples), dists

    def latentVisualize(
        self,
        input_features,
        sample_latent1=None,
        sample_latent2=None,
        sample_latent3=None,
    ):
        """
        Visualize the latent space by sampling or using provided latent variables.
        """
        dists = {}
        encoder_outs = {}
        samples = []

        # Downsampling
        encoder_outs["out1"] = self.down_blocks[0](input_features)
        encoder_outs["out2"] = F.dropout2d(
            self.down_blocks[1](encoder_outs["out1"]), p=0.5, training=self.training
        )
        encoder_outs["out3"] = F.dropout2d(
            self.down_blocks[2](encoder_outs["out2"]), p=0.3, training=self.training
        )
        encoder_outs["out4"], dists["dist1"] = self.down_blocks[3](encoder_outs["out3"])

        # Sampling and visualization
        for _ in range(self.num_samples):
            out = self.up_blocks[0](encoder_outs["out4"])
            latent1 = torch.nn.Upsample(
                size=encoder_outs["out3"].shape[2:], mode="nearest"
            )(dists["dist1"].rsample() if sample_latent1 is None else sample_latent1)
            out = torch.cat((encoder_outs["out3"], out, latent1), 1)

            out = F.dropout2d(out, p=0.5, training=self.training)
            out, dists["dist2"] = self.up_blocks[1](out)
            latent2 = torch.nn.Upsample(
                size=encoder_outs["out2"].shape[2:], mode="nearest"
            )(dists["dist2"].rsample() if sample_latent2 is None else sample_latent2)
            out = torch.cat((encoder_outs["out2"], out, latent2), 1)

            out = F.dropout2d(out, p=0.5, training=self.training)
            out, dists["dist3"] = self.up_blocks[2](out)
            latent3 = torch.nn.Upsample(
                size=encoder_outs["out1"].shape[2:], mode="nearest"
            )(dists["dist3"].rsample() if sample_latent3 is None else sample_latent3)
            out = torch.cat((encoder_outs["out1"], out, latent3), 1)

            segs = self.up_blocks[3](out)
            segs = self.crf(segs)
            segs = self.softmax(segs)

            samples.append(segs)

        return torch.stack(samples), dists
