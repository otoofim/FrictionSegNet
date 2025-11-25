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
                    input_dim=(base_channels * 4)
                    + (base_channels * 4)
                    + latent_var_size,  # skip + prev_out + latent
                    output_dim=base_channels * 2,
                    latent_dim=latent_var_size,
                    num_res_layers=num_res_layers,
                    activation=activation,
                ),
                UpConvBlock(
                    input_dim=(base_channels * 2)
                    + (base_channels * 2)
                    + latent_var_size,  # skip + prev_out + latent
                    output_dim=base_channels,
                    latent_dim=latent_var_size,
                    num_res_layers=num_res_layers,
                    activation=activation,
                ),
                UpConvBlock(
                    input_dim=(base_channels)
                    + (base_channels)
                    + latent_var_size,  # skip + prev_out + latent
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

        # Downsampling - process all blocks except potentially the last one
        for i, block in enumerate(self.down_blocks[:-1]):  # Exclude last block
            if i == 0:
                encoder_outs[f"out{i + 1}"] = block(input_features)
            else:
                encoder_outs[f"out{i + 1}"] = F.dropout2d(
                    block(encoder_outs[f"out{i}"]),
                    p=0.5 if i == 1 else 0.3,
                    training=self.training,
                )

        # Handle last down_block
        last_idx = len(self.down_blocks)
        if self.down_blocks[-1].latent_dim is not None:
            # Last block has latent_dim, so it returns (out, dist)
            encoder_outs[f"out{last_idx}"], dists[f"dist{dist_counter}"] = (
                self.down_blocks[-1](encoder_outs[f"out{last_idx - 1}"])
            )
            dist_counter += 1
        else:
            # Last block doesn't have latent_dim, apply dropout as usual
            encoder_outs[f"out{last_idx}"] = F.dropout2d(
                self.down_blocks[-1](encoder_outs[f"out{last_idx - 1}"]),
                p=0.3,
                training=self.training,
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

            # All up_blocks after the first need latent variable concatenated
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

            # Apply dropout and pass through block
            concatenated = F.dropout2d(concatenated, p=0.5, training=self.training)
            result = self.up_blocks[i](concatenated)

            # Handle output based on whether block has latent_dim
            if self.up_blocks[i].latent_dim is not None:
                out, dists[f"dist{dist_counter}"] = result
                dist_counter += 1
            else:
                out = result

        # Apply softmax to get probabilities
        segs = self.softmax(out)
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

                    # All up_blocks after the first need latent variable concatenated
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

                    # Pass through block
                    result = self.up_blocks[i](concatenated)

                    # Handle output based on whether block has latent_dim
                    if self.up_blocks[i].latent_dim is not None:
                        out, new_dist = result
                        if _ == 0:  # Only store dist on first sample
                            dists[f"dist{i + 1}"] = new_dist
                    else:
                        out = result

                # Apply softmax to get probabilities
                segs = self.softmax(out)
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
        dist_counter = 1

        # Downsampling - process all blocks except potentially the last one
        for i, block in enumerate(self.down_blocks[:-1]):  # Exclude last block
            if i == 0:
                encoder_outs[f"out{i + 1}"] = block(input_features)
            else:
                encoder_outs[f"out{i + 1}"] = F.dropout2d(
                    block(encoder_outs[f"out{i}"]),
                    p=0.5 if i == 1 else 0.3,
                    training=self.training,
                )

        # Handle last down_block
        last_idx = len(self.down_blocks)
        if self.down_blocks[-1].latent_dim is not None:
            # Last block has latent_dim, so it returns (out, dist)
            encoder_outs[f"out{last_idx}"], dists[f"dist{dist_counter}"] = (
                self.down_blocks[-1](encoder_outs[f"out{last_idx - 1}"])
            )
            dist_counter += 1
        else:
            # Last block doesn't have latent_dim, apply dropout as usual
            encoder_outs[f"out{last_idx}"] = F.dropout2d(
                self.down_blocks[-1](encoder_outs[f"out{last_idx - 1}"]),
                p=0.3,
                training=self.training,
            )

        # Custom latent samples (for visualization purposes)
        custom_latents = [sample_latent1, sample_latent2, sample_latent3]

        # Sampling and visualization
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

                # Use custom latent if provided, otherwise sample from distribution
                if custom_latents[i - 1] is not None:
                    latent = custom_latents[i - 1]
                else:
                    latent = dists[f"dist{i}"].rsample()

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

                # Apply dropout and pass through block
                concatenated = F.dropout2d(concatenated, p=0.5, training=self.training)
                result = self.up_blocks[i](concatenated)

                # Handle output based on whether block has latent_dim
                if self.up_blocks[i].latent_dim is not None:
                    out, new_dist = result
                    if _ == 0:  # Only store dist on first sample
                        dists[f"dist{i + 1}"] = new_dist
                else:
                    out = result

            # Apply softmax to get probabilities
            segs = self.softmax(out)
            samples.append(segs)

        return torch.stack(samples), dists
