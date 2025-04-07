from probabilistic_unet.train import train
from probabilistic_unet.utils import ConfigManager


if __name__ == "__main__":
    config = ConfigManager("config.yaml")
    train(config)
