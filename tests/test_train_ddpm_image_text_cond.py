import torch

import main
import scripts.train_ddpm_image_text_cond as trainer


class _DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        image = torch.zeros(3, 4, 4)
        return image, {"text": "hello world"}


class _DummyVQVAE(torch.nn.Module):
    def __init__(self, input_channels, VQVAE):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def to(self, device):
        self.param = self.param.to(device)
        return self

    def encode(self, images):
        batch = images.shape[0]
        latents = torch.zeros(batch, 2, 4, 4, device=images.device)
        return latents, None

    def decode(self, latents):
        return torch.zeros(latents.shape[0], 3, 8, 8, device=latents.device)

    def load_state_dict(self, state_dict, strict=True):
        return


class _DummyScheduler:
    def __init__(self, device, num_timesteps, beta_start, beta_end):
        self.device = device
        self.num_timesteps = num_timesteps
        self.sqrt_alphas_cumprod = torch.ones(num_timesteps, device=device)
        self.sqrt_one_minus_alphas_cumprod = torch.zeros(num_timesteps, device=device)

    def add_noise(self, latents, noise, timesteps):
        return latents

    def sample_prev_timestep(self, latents, noise_pred, timestep):
        return latents, latents


class _DummyUNet(torch.nn.Module):
    def __init__(self, UnetParams=None, **_kwargs):
        super().__init__()
        self.UnetParams = UnetParams
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.seen_cond_inputs = []

    def to(self, device):
        self.param = self.param.to(device)
        return self

    def train(self, mode=True):
        super().train(mode)
        return self

    def eval(self):
        super().eval()
        return self

    def forward(self, latents, timesteps, cond_input=None):
        self.seen_cond_inputs.append(cond_input)
        return latents + self.param


def test_train_adds_text_cond_to_unet(monkeypatch, tmp_path):
    # Ensure model/device utilities use CPU for the stubbed components
    trainer.DEVICE = torch.device("cpu")

    monkeypatch.setattr(trainer, "build_dataset", lambda *args, **kwargs: _DummyDataset())
    monkeypatch.setattr(trainer, "VQVAE", _DummyVQVAE)
    monkeypatch.setattr(trainer, "LinearNoiseScheduler", _DummyScheduler)

    created_unets = []
    monkeypatch.setattr(
        trainer,
        "UNet",
        lambda *args, **kwargs: created_unets.append(_DummyUNet(*args, **kwargs)) or created_unets[-1],
    )

    monkeypatch.setattr(
        trainer,
        "get_tokenizer_and_model",
        lambda *args, **kwargs: ("tokenizer", "model"),
    )
    monkeypatch.setattr(
        trainer,
        "get_text_representation",
        lambda text, *_args, **_kwargs: torch.ones(
            len(text) if isinstance(text, (list, tuple)) else 1, 2, 8
        ),
    )
    monkeypatch.setattr(
        trainer,
        "render_text_prompts",
        lambda prompts, size: torch.zeros((len(prompts), 3, size[0], size[1])),
    )
    monkeypatch.setattr(
        trainer,
        "format_image_condition",
        lambda _cond, batch, size: torch.zeros((batch, 3, size[0], size[1])),
    )

    saved = []

    def _record_save(images, directory, step, nrow, prefix):
        saved.append(
            {"count": int(images.shape[0]), "nrow": nrow, "prefix": prefix}
        )

    monkeypatch.setattr(trainer, "save_image_grid", _record_save)

    ckpt_dir = tmp_path / "celebhq"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "vqvae_autoencoder_ckpt.pth"
    torch.save({}, ckpt_path)

    config = {
        "diffusion_params": {"num_timesteps": 2, "beta_start": 0.1, "beta_end": 0.2},
        "dataset_params": {
            "name": "dummy",
            "im_channels": 3,
            "im_size": 4,
            "im_ext": "png",
            "im_path": str(tmp_path),
            "split": "train",
        },
        "UnetParams": {
            "down_channels": [2, 2],
            "mid_channels": [2, 2],
            "down_sample": [False],
            "attn_down": [False],
            "time_emb_dim": 4,
            "norm_channels": 4,
            "num_heads": 1,
            "conv_out_channels": 2,
            "num_down_layers": 1,
            "num_mid_layers": 1,
            "num_up_layers": 1,
            "im_channels": 3,
        },
        "VQVAE": {"z_channels": 2},
        "cond_model": {
            "condition_config": {
                "condition_types": ["text"],
                "text_condition_config": {
                    "text_embed_model": "clip",
                    "text_embed_dim": 8,
                    "cond_drop_prob": 0.0,
                },
            }
        },
        "train_params": {
            "seed": 0,
            "task_name": "unit_test",
            "task_name_ddpm_cond": str(tmp_path / "out"),
            "vqvae_model": str(ckpt_dir),
            "ldm_batch_size": 1,
            "ldm_img_save_steps": 1,
            "ldm_epochs": 1,
            "num_workers": 0,
            "num_samples": 1,
            "num_grid_rows": 1,
            "ldm_lr": 0.001,
            "ldm_ckpt_name": "ddpm_cond_ckpt.pth",
            "vqvae_autoencoder_ckpt_name": ckpt_path.name,
        },
    }

    trainer.train(config)

    assert created_unets, "UNet should be constructed during training"
    cond_inputs = created_unets[0].seen_cond_inputs
    assert cond_inputs, "UNet forward should receive conditioning input"
    assert "text" in cond_inputs[0], "Text conditioning must be passed to the model"
    assert cond_inputs[0]["text"].shape[0] == 1
    assert any(call["prefix"] == "viz" and call["nrow"] == 5 for call in saved)


def test_main_routes_image_text_mode(monkeypatch, tmp_path):
    called = {}

    def fake_train(cfg):
        called["train"] = cfg

    def fake_load(path):
        called["path"] = path
        return {"cfg": True}

    monkeypatch.setattr(main, "train_ddpm_image_text_cond", fake_train)
    monkeypatch.setattr(main, "load_ddpm_image_text_config", fake_load)

    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("placeholder")

    main.main(
        ["--mode", "ddpm_image_text_cond", "--config", str(cfg_path)]
    )

    assert called["path"] == cfg_path
    assert called["train"] == {"cfg": True}
