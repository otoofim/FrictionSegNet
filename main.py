# from probabilistic_unet.train import train
# from probabilistic_unet.utils.config_loader.config_manager import ConfigManager
from probabilistic_unet.model.prior import Prior
import torch
import torch.nn as nn

# if __name__ == "__main__":
#     config = ConfigManager("config.yml").get_configs()
#     print(config.project_name)
#     train(config)


if __name__ == "__main__":
    # Example usage

    
    model = Prior(num_samples=5, num_classes=3, latent_var_size=6, input_dim=3, base_channels=64, num_res_layers=2, activation=nn.ReLU)
    input_tensor = torch.randn(2, 3, 128, 128)  # Batch of 2 images
    # segs, dists = model(input_tensor)
    segs, dists = model.inference(input_tensor)
    print("Segmentation output shape:", segs.shape)