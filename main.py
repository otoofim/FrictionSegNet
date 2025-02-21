import argparse
import os
import sys
from frictionsegnet.Train import *
import yaml


def main():
    f = open("config.yml", "r")
    args = yaml.safe_load(f)
    train(**args)


if __name__ == "__main__":
    main()
