# Probabilistic U-Net

A PyTorch implementation of **Probabilistic U-Net** for semantic segmentation with uncertainty quantification. This project combines a prior and posterior network to learn a distribution over segmentation masks, enabling pixel-level uncertainty estimation through probabilistic sampling.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Theoretical Foundation](#theoretical-foundation)
- [Installation & Environment Setup](#installation--environment-setup)
- [Training](#training)
- [Model Architecture](#model-architecture)
- [Key Features](#key-features)

---

## Project Overview

The Probabilistic U-Net extends the classical U-Net architecture by incorporating a **Variational Autoencoder (VAE)** component in the latent space. This enables the model to:

- Generate multiple diverse segmentation masks for the same input image
- Estimate **aleatoric uncertainty** (data uncertainty) through sample diversity
- Learn **epistemic uncertainty** (model uncertainty) through the VAE posterior
- Provide probabilistic outputs rather than deterministic predictions

This is particularly valuable for:
- **Medical imaging** where multiple valid segmentations exist
- **Autonomous driving** where uncertainty estimation is critical for safety
- **Scene understanding** tasks with inherent ambiguity

### Use Case: Cityscapes Segmentation

This implementation is trained on the **Cityscapes dataset**, a large-scale urban scene understanding dataset with 30 semantic classes. The model learns to segment urban scenes while providing uncertainty estimates for each prediction.

---

## Theoretical Foundation

### Variational Autoencoder (VAE) Framework

The Probabilistic U-Net is built on the **Variational Autoencoder** framework:

$$p(y|x) = \int p(y|z, x) \cdot p(z|x) \, dz$$

where:
- $x$ is the input image
- $y$ is the segmentation mask
- $z$ is the latent variable

**Training objective (ELBO)**:
$$\mathcal{L} = \mathbb{E}_{q(z|x,y)}[\log p(y|z,x)] - \beta \cdot KL(q(z|x,y) \| p(z|x))$$

### Prior Network ($p(z|x)$)

The **prior network** learns the base distribution of segmentation masks conditioned only on the input image:

- **Architecture**: U-Net encoder-decoder with latent variable sampling at multiple levels
- **Output**: Probabilistic segmentation predictions $p(y|z,x)$
- **Sampling**: Generates multiple diverse segmentation samples from the latent distribution
- **Use case**: Inference and uncertainty estimation

### Posterior Network ($q(z|x,y)$)

The **posterior network** learns to refine the distribution when ground truth segmentation is available:

- **Input**: Image + ground truth segmentation (concatenated)
- **Architecture**: Similar to prior but with access to ground truth information
- **Output**: Posterior distribution approximating $q(z|x,y)$
- **Use case**: Training only (during validation/inference, only prior is used)

### ELBO Loss Components

The training loss consists of two parts:

1. **Reconstruction Loss** (Cross-Entropy):
   $$\mathcal{L}_{rec} = -\mathbb{E}_{q(z|x,y)}[\log p(y|z,x)]$$
   Ensures segmentation quality by minimizing classification error

2. **KL Divergence Regularization**:
   $$\mathcal{L}_{KL} = KL(q(z|x,y) \| p(z|x))$$
   Regularizes the latent space to stay close to a standard prior

3. **Total Loss**:
   $$\mathcal{L}_{total} = \mathcal{L}_{rec} + \beta \cdot \mathcal{L}_{KL}$$
   where $\beta$ is a hyperparameter controlling the trade-off

### Multi-Level Latent Variables

The model uses **hierarchical latent sampling** at three different decoder levels:

- **Level 1** (high-level semantic): Captures global scene structure
- **Level 2** (mid-level): Handles regional variations
- **Level 3** (low-level): Refines local details

This multi-scale approach improves both segmentation quality and uncertainty calibration.

### Uncertainty Quantification

The model provides two types of uncertainty estimates:

**Aleatoric Uncertainty** (Data Uncertainty):
- Captured through sample diversity
- Computed as the variance across multiple segmentation samples
- Reflects inherent ambiguity in the image

**Epistemic Uncertainty** (Model Uncertainty):
- Reflected in the posterior distribution
- Measured by calibration metrics (ECE, MCE)
- Indicates model confidence

---

## Installation & Environment Setup

### Prerequisites

- Python >= 3.12
- CUDA 11.8+ (for GPU support, optional but recommended)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Probabilistic-UNet
```

### Step 2: Install Dependencies

This project uses **UV** for fast and reliable dependency management.

#### Option A: Using UV (Recommended)

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install project dependencies**:
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows
   ```

#### Option B: Using pip with pyproject.toml

```bash
pip install -e .
```

### Step 3: Prepare Dataset

#### Cityscapes Dataset

1. **Download Cityscapes**:
   - Register at [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)
   - Download the following files:
     - `leftImg8bit_trainvaltest.zip` (leftImg8bit)
     - `gtFine_trainvaltest.zip` (gtFine)

2. **Extract to dataset directory**:
   ```bash
   unzip leftImg8bit_trainvaltest.zip -d datasets/Cityscapes/
   unzip gtFine_trainvaltest.zip -d datasets/Cityscapes/
   ```

3. **Directory structure**:
   ```
   datasets/Cityscapes/
   ├── leftImg8bit/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── gtFine/
       ├── train/
       ├── val/
       └── test/
   ```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root with your configuration:

```bash
# Dataset Configuration
DATASET_ROOT_DIR=./datasets/Cityscapes
DATASET_IMG_SIZE=512
DATASET_USE_AUGMENTATION=true
DATASET_BATCH_SIZE=8
DATASET_NUM_WORKERS=4

# Training Configuration
TRAINING_EPOCHS=100
TRAINING_BATCH_SIZE=8
TRAINING_LEARNING_RATE=0.001
TRAINING_OPTIMIZER=adamw
TRAINING_LR_SCHEDULER=cosine
TRAINING_PRECISION=16-mixed
TRAINING_SEED=42

# Model Configuration
MODEL_LATENT_DIM=6
MODEL_BETA=5.0
MODEL_USE_POSTERIOR=true
MODEL_NUM_SAMPLES=16

# Logging & Checkpointing
TRAINING_CHECKPOINT_DIR=./checkpoints
TRAINING_LOG_EVERY_N_STEPS=50

# Weights & Biases (Optional)
WANDB_PROJECT=probabilistic-unet
WANDB_ENTITY=<your-username>
WANDB_API_KEY=<your-api-key>
```

**Key Configuration Parameters**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_IMG_SIZE` | 512 | Input image resolution |
| `DATASET_USE_AUGMENTATION` | true | Enable data augmentation |
| `TRAINING_EPOCHS` | 100 | Number of training epochs |
| `MODEL_LATENT_DIM` | 6 | Latent space dimensionality |
| `MODEL_BETA` | 5.0 | KL divergence weight (higher = stronger regularization) |
| `MODEL_NUM_SAMPLES` | 16 | Samples to generate during inference |
| `TRAINING_PRECISION` | 16-mixed | Mixed precision training for faster computation |

---

## Training

### Quick Start

Run the training script with default configuration:

```bash
uv run python main.py
```

### Training Script Details

The `main.py` script:
1. Loads configuration from `.env` file
2. Initializes the Cityscapes dataset and dataloaders
3. Creates the Probabilistic U-Net model
4. Starts training using PyTorch Lightning
5. Logs metrics to Weights & Biases (if configured)
6. Saves model checkpoints

### Training Output

During training, you'll see:

```
Probabilistic U-Net Training with PyTorch Lightning
================================================================================
Random seed set to 42
Initializing data module...
Initializing model...
Initialized Probabilistic U-Net with 30 classes
Latent dimension: 6
Beta (KL weight): 5.0
Number of samples: 16

Trainer configured:
  - Max epochs: 100
  - Accelerator: gpu
  - Precision: 16-mixed
  - Gradient clip: 1.0
  - Accumulate grad batches: 1

Starting training...
```

### Monitoring Training

#### Using Weights & Biases

If you've configured WandB credentials, metrics are automatically logged to [wandb.ai](https://wandb.ai/):

- **Loss curves**: Total loss, reconstruction loss, KL divergence
- **Metrics**: mIoU (mean Intersection over Union) per epoch
- **Calibration**: Expected Calibration Error (ECE) for uncertainty quality
- **Predictions**: Sample predictions visualized on validation set
- **Confusion matrices**: Per-class performance analysis

#### View Checkpoints

Trained model checkpoints are saved in `./checkpoints/`:

```bash
ls -la checkpoints/
# probabilistic-unet-epoch=00-val_mIoU=0.0059.ckpt
# probabilistic-unet-epoch=01-val_mIoU=0.0175.ckpt
# last.ckpt
```

### Training Hyperparameters

Key hyperparameters and their effects:

**Model Architecture**:
- `LATENT_DIM=6`: Affects model capacity and diversity of generated samples
- `BETA=5.0`: Controls uncertainty calibration (higher = more uncertainty)
- `NUM_SAMPLES=16`: Samples per image during inference (higher = better estimates, slower)

**Optimization**:
- `LEARNING_RATE=0.001`: Initial learning rate
- `OPTIMIZER=adamw`: Adam with weight decay (recommended)
- `LR_SCHEDULER=cosine`: Cosine annealing (smooth decay)

**Training**:
- `BATCH_SIZE=8`: Increase for faster training (if memory allows)
- `PRECISION=16-mixed`: Mixed precision for ~2x faster training with minimal quality loss
- `EPOCHS=100`: Number of passes through dataset

### Tips for Training

1. **GPU Memory Issues**: Reduce `BATCH_SIZE` or `MODEL_NUM_SAMPLES`
2. **Slow Training**: Enable `PRECISION=16-mixed` for ~2x speedup
3. **Poor Uncertainty**: Increase `MODEL_BETA` to 10-20
4. **Overfitting**: Enable `DATASET_USE_AUGMENTATION=true` and tune `TRAINING_EARLY_STOP_PATIENCE`

### Resume Training

To resume from a checkpoint:

```bash
# Set in .env
TRAINING_RESUME_FROM_CHECKPOINT=./checkpoints/last.ckpt
```

Then run:
```bash
uv run python main.py
```

---

## References

1. **Kohl et al., 2018** - Probabilistic U-Net: [A probabilistic u-net for segmentation of ambiguous images](https://arxiv.org/abs/1806.05034)

2. **Otoofi et al., 2024** - FrictionSegNet: [Simultaneous semantic segmentation and friction estimation using hierarchical latent variable models](https://repository.lboro.ac.uk/articles/journal_contribution/FrictionSegNet_simultaneous_semantic_segmentation_and_friction_estimation_using_hierarchical_latent_variable_models/27794922)

3. **Otoofi, 2024** - PhD Thesis: [Simultaneous semantic segmentation and friction estimation using hierarchical latent variable models](https://repository.lboro.ac.uk/articles/thesis/Simultaneous_semantic_segmentation_and_friction_estimation_using_hierarchical_latent_variable_models/28352609)

4. **Cordts et al., 2016** - Cityscapes Dataset: [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://www.cityscapes-dataset.com/)

5. **Kingma & Welling, 2013** - Variational Inference: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

6. **Falcon et al., 2019** - PyTorch Lightning: [PyTorch Lightning](https://lightning.ai/)

---

## Project Structure

```
Probabilistic-UNet/
├── main.py                          # Entry point for training
├── pyproject.toml                   # Project dependencies
├── README.md                        # This file
├── .env                             # Configuration (create this)
├── checkpoints/                     # Saved model checkpoints
├── datasets/                        # Dataset directory
├── logs/                            # Training logs
├── probabilistic_unet/
│   ├── train_lightning.py          # Lightning training module
│   ├── model/
│   │   ├── pro_unet.py             # Main Probabilistic U-Net class
│   │   ├── prior.py                # Prior network
│   │   ├── posterior.py            # Posterior network
│   │   ├── unet_blocks.py          # U-Net building blocks
│   │   └── residual_block.py       # Residual blocks
│   ├── dataloader/
│   │   ├── cityscapes_loader.py    # Cityscapes data loading
│   │   ├── lightning_dataloader.py # Lightning data module
│   │   └── base_segmentation_dataset.py
│   └── utils/
│       ├── logger.py               # Logging utilities
│       ├── config_loader/          # Configuration management
│       ├── objective_functions/    # Loss functions
│       ├── confusion_matrix/       # Metrics computation
│       └── support_functions/      # Helper functions
└── wandb/                          # Weights & Biases logs
```

---

## License

[Add your license here]

## Citation

If you use this implementation in your research, please cite the original Probabilistic U-Net paper along with the related publications:

**Original Probabilistic U-Net**:
```bibtex
@article{kohl2018probabilistic,
  title={A probabilistic u-net for segmentation of ambiguous images},
  author={Kohl, Simon and Romijn, Bas and Reyes, Mauricio},
  journal={arXiv preprint arXiv:1806.05034},
  year={2018}
}
```

**Related Paper**:
```bibtex
@article{otoofi2024frictionsegnet,
  title={FrictionSegNet: Simultaneous Semantic Segmentation and Friction Estimation Using Hierarchical Latent Variable Models},
  author={Otoofi, Mohammad and Laine, Leo and Henderson, Leon and Midgley, William JB and Justham, Laura and Fleming, James},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024},
  publisher={IEEE}
}
```

**PhD Thesis**:
```bibtex
@phdthesis{otoofisimultaneous,
  title={Simultaneous semantic segmentation and friction estimation using hierarchical latent variable models},
  author={Otoofi, Mohammad},
  school={Loughborough University},
  year={2024}
}
```
