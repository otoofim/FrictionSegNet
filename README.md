# Probabilistic U-Net

A specialized implementation of probabilistic U-Net for semantic segmentation with VAE latent space sampling.

## Package Management

This project now uses **UV** instead of Poetry for faster and more reliable dependency management.

### Installation

1. Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install project dependencies:
```bash
uv sync
```

### Usage

Run scripts using UV:
```bash
uv run python main.py
uv run python probabilistic_unet/train.py
```

Activate the virtual environment:
```bash
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Key Features Preserved

✅ **VAE Latent Space Sampling**: The specialized VAE implementation with `rsample()` and `sample()` methods is fully preserved  
✅ **Multi-level Latent Variables**: Three-level latent distribution hierarchy (dist1, dist2, dist3)  
✅ **Probabilistic U-Net Architecture**: Complete prior and posterior networks with custom sampling strategies  
✅ **Uncertainty Quantification**: Probabilistic predictions for semantic segmentation with uncertainty estimation  

### Migration from Poetry

The project has been successfully migrated from Poetry to UV while preserving all functionality:

- ✅ All dependencies resolved and installed
- ✅ VAE latent space sampling implementation intact
- ✅ Probabilistic U-Net architecture preserved
- ✅ Custom loss functions and GECO implementation maintained

### Dependencies

Key dependencies include:
- PyTorch >= 2.5.1
- TorchVision >= 0.20.1
- NumPy >= 1.26.0
- TorchMetrics >= 1.7.0
- CRFSeg >= 1.0.0
- Weights & Biases for experiment tracking

For the complete list, see `pyproject.toml`.