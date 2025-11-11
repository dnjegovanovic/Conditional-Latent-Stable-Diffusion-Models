import argparse
from pathlib import Path

from scripts.train_ddpm_cond import main as train_ddpm_conditional
from scripts.train_ddpm_vqvae import train as train_ddpm_vqvae


def parse_args():
    parser = argparse.ArgumentParser(description="Select training routine")
    parser.add_argument(
        "--mode",
        choices=["ddpm_cond", "ddpm_vqvae"],
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
        from scripts.train_ddpm_vqvae import load_config

        train_ddpm_vqvae(load_config(config_path))
    else:
        train_ddpm_conditional()
