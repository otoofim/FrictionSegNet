from probabilistic_unet.train import train
from probabilistic_unet.utils.config_loader.config_manager import ConfigManager


if __name__ == "__main__":
    config = ConfigManager("config.yml").get_configs()
    print(config.project_name)
    train(config)
