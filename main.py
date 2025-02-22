from probabilistic_unet.Train import train
from probabilistic_unet.utils import ConfigManager


if __name__ == "__main__":
    config = ConfigManager("config.yaml")
    train(config)
