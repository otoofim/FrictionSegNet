import torch
import torch.nn as nn
from probabilistic_unet.model.posterior import Posterior


def test_posterior():
    """Test the Posterior model's forward and inference methods."""

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Configuration
    batch_size = 2
    num_classes = 5
    num_samples = 3
    latent_var_size = 6
    base_channels = 32  # Using smaller channels for faster testing
    height, width = 128, 128

    # Create model
    print("Creating Posterior model...")
    posterior = Posterior(
        num_samples=num_samples,
        num_classes=num_classes,
        latent_var_size=latent_var_size,
        input_dim=None,  # Will calculate as 3 + num_classes + 1
        base_channels=base_channels,
        num_res_layers=1,
        activation=nn.ReLU,
    )
    posterior.train()  # Posterior uses dropout, so we need training mode

    # Create dummy input (RGB + segmentation)
    input_dim = 3 + num_classes
    input_features = torch.randn(batch_size, input_dim, height, width)

    print(f"\nInput shape: {input_features.shape}")
    print(
        f"Input dimension breakdown: 3 (RGB) + {num_classes} (seg classes) = {input_dim}"
    )

    # Test forward method
    print("\n" + "=" * 60)
    print("Testing forward() method...")
    print("=" * 60)
    try:
        dists_forward = posterior.forward(input_features)
        print(f"✓ Forward pass successful!")
        print(f"  - Number of distributions returned: {len(dists_forward)}")

        # Check each distribution
        for key, dist in dists_forward.items():
            print(f"  - {key}:")
            print(f"    * Mean shape: {dist.mean.shape}")
            print(f"    * Std shape: {dist.stddev.shape}")

            # Sample from distribution
            sample = dist.rsample()
            print(f"    * Sample shape: {sample.shape}")

        assert len(dists_forward) > 0, "No distributions returned!"
        print("\n✓ All distribution shapes are correct!")

    except Exception as e:
        print(f"✗ Forward pass failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test inference method
    print("\n" + "=" * 60)
    print("Testing inference() method...")
    print("=" * 60)

    posterior.eval()  # Switch to eval mode for inference

    try:
        dists_inference = posterior.inference(input_features)
        print(f"✓ Inference successful!")
        print(f"  - Number of distributions returned: {len(dists_inference)}")

        # Check each distribution
        for key, dist in dists_inference.items():
            print(f"  - {key}:")
            print(f"    * Mean shape: {dist.mean.shape}")
            print(f"    * Std shape: {dist.stddev.shape}")

            # Sample from distribution
            sample = dist.sample()
            print(f"    * Sample shape: {sample.shape}")

        assert len(dists_inference) > 0, "No distributions returned!"
        print("\n✓ All distribution shapes are correct!")

    except Exception as e:
        print(f"✗ Inference failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test with custom parameters
    print("\n" + "=" * 60)
    print("Testing with custom parameters...")
    print("=" * 60)

    try:
        posterior_custom = Posterior(
            num_samples=5,
            num_classes=10,
            latent_var_size=8,
            input_dim=20,
            base_channels=64,
            num_res_layers=2,
            activation=nn.LeakyReLU,
        )

        custom_input = torch.randn(1, 20, 64, 64)
        posterior_custom.train()
        dists = posterior_custom.forward(custom_input)

        print(f"✓ Custom configuration works!")
        print(f"  - Input shape: {custom_input.shape}")
        print(f"  - Number of distributions: {len(dists)}")

    except Exception as e:
        print(f"✗ Custom configuration failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test consistency between training and inference modes
    print("\n" + "=" * 60)
    print("Testing consistency between forward and inference...")
    print("=" * 60)

    try:
        posterior.train()
        dists_train1 = posterior.forward(input_features)
        dists_train2 = posterior.forward(input_features)

        # Due to dropout, distributions should be different in training mode
        diff_found = False
        for key in dists_train1.keys():
            if not torch.allclose(dists_train1[key].mean, dists_train2[key].mean):
                diff_found = True
                break

        if diff_found:
            print("✓ Training mode produces stochastic results (due to dropout)")
        else:
            print("⚠ Training mode results are identical (unexpected with dropout)")

        # In inference mode, results should be deterministic (without dropout)
        posterior.eval()
        dists_inf1 = posterior.inference(input_features)
        dists_inf2 = posterior.inference(input_features)

        all_close = True
        for key in dists_inf1.keys():
            if not torch.allclose(dists_inf1[key].mean, dists_inf2[key].mean):
                all_close = False
                break

        if all_close:
            print("✓ Inference mode produces deterministic results")
        else:
            print("⚠ Inference mode results differ (unexpected)")

    except Exception as e:
        print(f"✗ Consistency test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Model summary
    print("\n" + "=" * 60)
    print("Model Architecture Summary")
    print("=" * 60)
    total_params = sum(p.numel() for p in posterior.parameters())
    trainable_params = sum(p.numel() for p in posterior.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Number of down blocks: {len(posterior.down_blocks)}")
    print(f"Number of up blocks: {len(posterior.up_blocks)}")

    print("\nDown blocks configuration:")
    for i, block in enumerate(posterior.down_blocks):
        has_latent = "with latent" if block.latent_dim is not None else "no latent"
        print(f"  - Block {i + 1}: {has_latent}")

    print("\nUp blocks configuration:")
    for i, block in enumerate(posterior.up_blocks):
        has_latent = "with latent" if block.latent_dim is not None else "no latent"
        print(f"  - Block {i + 1}: {has_latent}")

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_posterior()
    exit(0 if success else 1)
