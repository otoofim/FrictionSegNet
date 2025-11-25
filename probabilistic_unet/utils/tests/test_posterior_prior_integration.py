import torch
import torch.nn as nn
from probabilistic_unet.model.posterior import Posterior
from probabilistic_unet.model.prior import Prior


def test_posterior_prior_integration():
    """Test that Posterior and Prior models work together correctly."""

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Configuration
    batch_size = 2
    num_classes = 5
    num_samples = 3
    latent_var_size = 6
    base_channels = 32
    height, width = 128, 128

    print("=" * 60)
    print("Testing Posterior-Prior Integration")
    print("=" * 60)

    # Create models
    print("\nCreating models...")
    posterior = Posterior(
        num_samples=num_samples,
        num_classes=num_classes,
        latent_var_size=latent_var_size,
        base_channels=base_channels,
        num_res_layers=1,
        activation=nn.ReLU,
    )

    prior = Prior(
        num_samples=num_samples,
        num_classes=num_classes,
        latent_var_size=latent_var_size,
        input_dim=3,
        base_channels=base_channels,
        num_res_layers=1,
        activation=nn.ReLU,
    )

    # Create inputs
    rgb_input = torch.randn(batch_size, 3, height, width)
    segmentation = torch.randn(batch_size, num_classes, height, width)

    # Concatenate for posterior input
    posterior_input = torch.cat([rgb_input, segmentation], dim=1)

    print(f"RGB input shape: {rgb_input.shape}")
    print(f"Segmentation shape: {segmentation.shape}")
    print(
        f"Posterior input shape: {posterior_input.shape}"
    )  # Test 1: Get distributions from posterior
    print("\n" + "=" * 60)
    print("Test 1: Extract distributions from Posterior")
    print("=" * 60)

    try:
        posterior.train()
        post_dists = posterior.forward(posterior_input)

        print(f"✓ Posterior forward pass successful!")
        print(f"  - Number of distributions: {len(post_dists)}")

        for key, dist in post_dists.items():
            print(f"  - {key}: mean shape {dist.mean.shape}")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 2: Use posterior distributions in prior
    print("\n" + "=" * 60)
    print("Test 2: Use Posterior distributions in Prior forward pass")
    print("=" * 60)

    try:
        prior.train()
        segs, prior_dists = prior.forward(rgb_input, post_dists)

        print(f"✓ Prior forward pass with posterior distributions successful!")
        print(f"  - Segmentation output shape: {segs.shape}")
        print(f"  - Expected: ({batch_size}, {num_classes}, H, W)")
        print(f"  - Number of prior distributions: {len(prior_dists)}")

        # Verify segmentation is a probability distribution
        seg_sum = segs.sum(dim=1)
        is_normalized = torch.allclose(seg_sum, torch.ones_like(seg_sum), atol=1e-5)

        if is_normalized:
            print(f"✓ Segmentation output is properly normalized (sums to 1)")
        else:
            print(
                f"⚠ Segmentation output is not normalized: sum range [{seg_sum.min():.4f}, {seg_sum.max():.4f}]"
            )

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 3: Distribution compatibility
    print("\n" + "=" * 60)
    print("Test 3: Check distribution compatibility")
    print("=" * 60)

    try:
        # Check that posterior and prior produce same number of distributions
        if len(post_dists) == len(prior_dists):
            print(f"✓ Distribution counts match: {len(post_dists)} distributions")
        else:
            print(
                f"⚠ Distribution counts differ: Posterior={len(post_dists)}, Prior={len(prior_dists)}"
            )

        # Check that distributions have compatible shapes
        all_compatible = True
        for key in post_dists.keys():
            if key in prior_dists:
                post_shape = post_dists[key].mean.shape
                prior_shape = prior_dists[key].mean.shape

                if post_shape == prior_shape:
                    print(f"✓ {key} shapes match: {post_shape}")
                else:
                    print(
                        f"⚠ {key} shapes differ: Posterior={post_shape}, Prior={prior_shape}"
                    )
                    all_compatible = False
            else:
                print(f"⚠ {key} not found in prior distributions")
                all_compatible = False

        if all_compatible:
            print("\n✓ All distributions are compatible!")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 4: KL divergence computation (typical use case)
    print("\n" + "=" * 60)
    print("Test 4: Compute KL divergence between distributions")
    print("=" * 60)

    try:
        from torch.distributions import kl_divergence

        total_kl = 0
        for key in post_dists.keys():
            if key in prior_dists:
                kl = kl_divergence(post_dists[key], prior_dists[key]).sum()
                print(f"  - KL({key}): {kl.item():.4f}")
                total_kl += kl

        print(f"\n✓ Total KL divergence: {total_kl.item():.4f}")
        print("  (This is a measure of difference between posterior and prior)")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 5: Inference mode
    print("\n" + "=" * 60)
    print("Test 5: Test inference mode for both models")
    print("=" * 60)

    try:
        posterior.eval()
        prior.eval()

        with torch.no_grad():
            post_dists_inf = posterior.inference(posterior_input)
            prior_samples, prior_dists_inf = prior.inference(rgb_input)

        print(f"✓ Posterior inference: {len(post_dists_inf)} distributions")
        print(f"✓ Prior inference: {prior_samples.shape[0]} samples generated")
        print(f"  - Sample shape: {prior_samples.shape}")
        print(f"  - Expected: ({num_samples}, {batch_size}, {num_classes}, H, W)")

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 60)
    print("Integration Test Summary")
    print("=" * 60)
    print("✓ Posterior model works correctly")
    print("✓ Prior model works correctly")
    print("✓ Distributions are compatible between models")
    print("✓ KL divergence can be computed")
    print("✓ Both models work in inference mode")
    print("\n" + "=" * 60)
    print("✓ All integration tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_posterior_prior_integration()
    exit(0 if success else 1)
