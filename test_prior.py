import torch
import torch.nn as nn
from probabilistic_unet.model.prior import Prior
# from probabilistic_unet.model.posterior import Posterior


def test_prior():
    """Test the Prior model's forward and inference methods."""

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Configuration
    batch_size = 2
    num_classes = 5
    num_samples = 3
    latent_var_size = 6
    input_dim = 3
    base_channels = 32  # Using smaller channels for faster testing
    height, width = 128, 128

    # Create model
    print("Creating Prior model...")
    prior = Prior(
        num_samples=num_samples,
        num_classes=num_classes,
        latent_var_size=latent_var_size,
        input_dim=input_dim,
        base_channels=base_channels,
        num_res_layers=1,
        activation=nn.ReLU,
    )
    prior.eval()

    # Create dummy input
    input_features = torch.randn(batch_size, input_dim, height, width)

    print(f"\nInput shape: {input_features.shape}")

    # Test inference method
    print("\n" + "=" * 60)
    print("Testing inference() method...")
    print("=" * 60)
    try:
        samples, dists_inference = prior.inference(input_features)
        print(f"✓ Inference successful!")
        print(f"  - Output samples shape: {samples.shape}")
        print(f"  - Expected: ({num_samples}, {batch_size}, {num_classes}, H, W)")
        print(f"  - Number of distributions: {len(dists_inference)}")

        # Check shapes
        assert samples.shape[0] == num_samples, (
            f"Expected {num_samples} samples, got {samples.shape[0]}"
        )
        assert samples.shape[1] == batch_size, (
            f"Expected batch size {batch_size}, got {samples.shape[1]}"
        )
        assert samples.shape[2] == num_classes, (
            f"Expected {num_classes} classes, got {samples.shape[2]}"
        )
        print(f"✓ All shape assertions passed!")

    except Exception as e:
        print(f"✗ Inference failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test forward method
    print("\n" + "=" * 60)
    print("Testing forward() method...")
    print("=" * 60)

    # Create dummy posterior distributions for forward pass
    # We'll use the distributions from inference as mock posterior
    print("Creating mock posterior distributions...")
    post_dist = {}
    for key, dist in dists_inference.items():
        # Create a simple mock distribution with sample and rsample methods
        post_dist[key] = dist

    try:
        prior.train()  # Switch to training mode to test dropout
        segs, dists_forward = prior.forward(input_features, post_dist)
        print(f"✓ Forward pass successful!")
        print(f"  - Output segmentation shape: {segs.shape}")
        print(f"  - Expected: ({batch_size}, {num_classes}, H, W)")
        print(f"  - Number of distributions: {len(dists_forward)}")

        # Check shapes
        assert segs.shape[0] == batch_size, (
            f"Expected batch size {batch_size}, got {segs.shape[0]}"
        )
        assert segs.shape[1] == num_classes, (
            f"Expected {num_classes} classes, got {segs.shape[1]}"
        )
        print(f"✓ All shape assertions passed!")

        # Check that output is a valid probability distribution
        print(f"\nChecking probability distribution properties...")
        print(f"  - Min value: {segs.min().item():.6f}")
        print(f"  - Max value: {segs.max().item():.6f}")
        print(
            f"  - Sum along class dimension (should be ~1.0): {segs.sum(dim=1).mean().item():.6f}"
        )

        assert segs.min() >= 0, "Output contains negative values!"
        assert segs.max() <= 1, "Output contains values > 1!"
        assert torch.allclose(
            segs.sum(dim=1),
            torch.ones(batch_size, segs.shape[2], segs.shape[3]),
            atol=1e-5,
        ), "Output doesn't sum to 1 along class dimension!"
        print(f"✓ Output is a valid probability distribution!")

    except Exception as e:
        print(f"✗ Forward pass failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test latentVisualize method
    print("\n" + "=" * 60)
    print("Testing latentVisualize() method...")
    print("=" * 60)

    try:
        prior.eval()
        samples_viz, dists_viz = prior.latentVisualize(input_features)
        print(f"✓ latentVisualize successful!")
        print(f"  - Output samples shape: {samples_viz.shape}")
        print(f"  - Expected: ({num_samples}, {batch_size}, {num_classes}, H, W)")
        print(f"  - Number of distributions: {len(dists_viz)}")

        # Check shapes
        assert samples_viz.shape[0] == num_samples, (
            f"Expected {num_samples} samples, got {samples_viz.shape[0]}"
        )
        assert samples_viz.shape[1] == batch_size, (
            f"Expected batch size {batch_size}, got {samples_viz.shape[1]}"
        )
        assert samples_viz.shape[2] == num_classes, (
            f"Expected {num_classes} classes, got {samples_viz.shape[2]}"
        )
        print(f"✓ All shape assertions passed!")

        # Test with custom latent samples
        print(f"\nTesting with custom latent samples...")
        custom_latent = dists_viz["dist1"].sample()
        samples_custom, _ = prior.latentVisualize(
            input_features, sample_latent1=custom_latent
        )
        print(f"✓ Custom latent visualization successful!")
        print(f"  - Output shape: {samples_custom.shape}")

    except Exception as e:
        print(f"✗ latentVisualize failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_prior()
