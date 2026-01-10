import sys
from pathlib import Path

import main


MODES = [
    (
        "vqvae",
        "VQ-VAE autoencoder",
        "con_stable_diff_model/config_vqvae_celeb.yml",
    ),
    (
        "ddpm_vqvae",
        "Latent DDPM (unconditional)",
        "con_stable_diff_model/config_vqvae_celeb.yml",
    ),
    (
        "ddpm_cond",
        "Conditional DDPM (class/image/text)",
        "con_stable_diff_model/config_celeb.yml",
    ),
    (
        "ddpm_image_text_cond",
        "Conditional DDPM (text + image)",
        "con_stable_diff_model/config_celeb_text_image.yml",
    ),
]


def _print_header(title: str) -> None:
    line = "=" * len(title)
    print(line)
    print(title)
    print(line)


def _prompt_choice(prompt: str, max_choice: int, default: int) -> int:
    while True:
        value = input(prompt).strip().lower()
        if value in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if value == "":
            return default
        if value.isdigit():
            choice = int(value)
            if 1 <= choice <= max_choice:
                return choice
        print(f"Please enter a number between 1 and {max_choice}, or 'q' to exit.")


def _list_configs(config_dir: Path) -> list[Path]:
    if not config_dir.exists():
        return []
    return sorted(p for p in config_dir.glob("*.yml") if p.is_file())


def _choose_config(default_path: Path, config_dir: Path) -> Path:
    configs = _list_configs(config_dir)
    default_idx = None
    for idx, path in enumerate(configs, start=1):
        if path == default_path:
            default_idx = idx
            break

    print("Available config files:")
    if configs:
        for idx, path in enumerate(configs, start=1):
            marker = " (default)" if default_idx == idx else ""
            print(f"{idx}) {path}{marker}")
    else:
        print("  (no config files found)")

    print("m) enter path manually")
    prompt = "Select config [default: {}]: ".format(
        default_idx if default_idx is not None else "m"
    )

    while True:
        choice = input(prompt).strip().lower()
        if choice in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if choice == "" and default_idx is not None:
            return default_path
        if choice in {"m", ""}:
            manual = input("Enter config path: ").strip()
            if manual:
                return Path(manual).expanduser()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(configs):
                return configs[idx - 1]
        print("Please select a listed config number, or 'm' for manual path.")


def run_cli() -> None:
    _print_header("Conditional Latent Diffusion Training CLI")
    print("Select a training mode:")
    for idx, (mode, desc, _default_cfg) in enumerate(MODES, start=1):
        print(f"{idx}) {mode:<20} {desc}")

    choice = _prompt_choice(
        f"Enter choice [1-{len(MODES)}] (default 1): ", len(MODES), 1
    )
    mode, desc, default_cfg = MODES[choice - 1]
    print(f"Selected: {mode} - {desc}")

    default_path = Path(default_cfg)
    config_path = _choose_config(default_path, Path("con_stable_diff_model"))

    print("\nRun configuration:")
    print(f"  mode:   {mode}")
    print(f"  config: {config_path}")
    confirm = input("Run now? [y/N]: ").strip().lower()
    if confirm not in {"y", "yes"}:
        print("Cancelled.")
        return

    main.main(["--mode", mode, "--config", str(config_path)])


if __name__ == "__main__":
    try:
        run_cli()
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
