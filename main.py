import argparse
from pathlib import Path

from scripts.train_ddpm_cond import (
    train as train_ddpm_cond,
    load_config as load_ddpm_cond_config,
)
from scripts.train_ddpm_vqvae import (
    train as train_ddpm_vqvae,
    load_config as load_ddpm_vqvae_config,
)
from scripts.train_vqvae import train as train_vqvae, load_config as load_vqvae_config


def parse_args():
    parser = argparse.ArgumentParser(description="Select training routine")
    parser.add_argument(
        "--mode",
        choices=["ddpm_cond", "ddpm_vqvae", "vqvae"],
        default="ddpm_cond",
        help="Which trainer to run.",
    )
    parser.add_argument(
        "--config",
        default="./con_stable_diff_model/config.yml",
        help="Path to the configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)

    if args.mode == "ddpm_vqvae":
        train_ddpm_vqvae(load_ddpm_vqvae_config(config_path))
    elif args.mode == "vqvae":
        train_vqvae(load_vqvae_config(config_path))
    else:
        train_ddpm_cond(load_ddpm_cond_config(config_path))
