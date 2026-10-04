"""Hyper1D model boundary, evaluation and timed-training integration."""

from collections import defaultdict
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import pytest
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from globalsplat.compression import initialize_observable_from_vanilla
from globalsplat.model.globalsplat import GlobalSplat
from globalsplat.model.model_wrapper import GlobalSplatModule
from globalsplat.model.optim import build_optimizer_and_scheduler
from globalsplat.pilot import InvocationTimer, configure_fixed_validation
from globalsplat.dataset.data_module import UpstreamBackedDataModule

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_hyper1d", ROOT / "scripts/run_hyper1d.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(77)
    yield
    torch.set_num_threads(previous)


def small_model(codec=True, architecture="legacy", paths=1, use_morton=False):
    return GlobalSplat(static_only=True, sh_degree=0, patch_size=4,
        latent_rep_token_amount=33, dim_latents=32, dim_rays=16, dim_rgb_feat=16,
        rounds=1, slot_calib_layers_per_round=1, num_heads=4, M_max=2,
        freeze_globalsplat=codec,
        feature_codec=dict(codec_type="hyper1d", geometry_observable_channels=28,
            n=3, m=6, adapter_hidden=3, architecture=architecture,
            paths=paths, use_morton=use_morton) if codec else None)


def sample_batch():
    cameras = torch.eye(4).repeat(1, 2, 1, 1)
    cameras[:, 1, 0, 3] = 0.1
    intrinsics = torch.tensor([[8., 0., 4.], [0., 8., 4.], [0., 0., 1.]]).repeat(1, 2, 1, 1)
    inputs = dict(images=torch.rand(1, 2, 3, 8, 8), intrinsic=intrinsics,
                  c2w=cameras, frame_ids=torch.tensor([[1, 7]]))
    return {"inputs": inputs, "targets": dict(inputs, frame_ids=torch.tensor([[3, 5]])),
            "scene_info": {"scene": ["fixed_scene"]}}


@pytest.mark.parametrize("architecture", ["legacy", "plain4"])
def test_qr_boundary_freezing_and_full_model_gradients(monkeypatch, architecture):
    source = small_model(False).eval()
    model = small_model(True, architecture).train()
    report = initialize_observable_from_vanilla(model, {"state_dict": source.state_dict()})
    assert report.max_reparameterization_error < 1e-6
    t, g = torch.randn(1, 33, 32), torch.randn(1, 33, 32)
    before = source.gaussian_decoder((t, g))
    after = model.gaussian_decoder((t, model.feature_codec.project_geometry(g)))
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-5)
    monkeypatch.setattr(model.gaussian_decoder, "decoded_token_centers",
                        lambda *a: pytest.fail("Morton OFF must skip center decoding"))
    gaussians = model(sample_batch()["inputs"])
    loss = gaussians.means.square().mean() + gaussians.sh.square().mean() + model.last_codec_output.estimated_bits / 10000
    loss.backward()
    assert all(not p.requires_grad and p.grad is None for name, p in model.named_parameters() if not name.startswith("feature_codec."))
    assert model.feature_codec.geometry_projection.weight.grad.abs().sum() > 0
    model.eval()
    model.feature_codec.update()
    compressed = model.compress_scene_tokens((t, g))
    result = model.decode_scene_tokens(model.decompress_scene_tokens(compressed.data))
    assert result.num_gaussians == 33


@pytest.mark.parametrize("architecture", ["legacy", "plain4"])
@pytest.mark.parametrize("paths", [1, 2])
def test_checkpoint_save_load_no_score_attributes(architecture, paths):
    model = small_model(architecture=architecture, paths=paths)
    model.feature_codec.update()
    module = GlobalSplatModule(model, eval_mode=True)
    module.hyper1d_provenance = {"source": "test"}
    checkpoint = {"state_dict": module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["feature_codec_config"]["codec_type"] == "hyper1d"
    assert "score_mean_offset_enabled" not in checkpoint
    fresh = GlobalSplatModule(small_model(architecture=architecture, paths=paths), eval_mode=True)
    fresh.on_load_checkpoint(checkpoint)
    fresh.load_state_dict(checkpoint["state_dict"], strict=True)
    assert fresh.hyper1d_provenance == module.hyper1d_provenance
    torch.testing.assert_close(fresh.model.feature_codec.entropy_bottleneck._quantized_cdf,
                               model.feature_codec.entropy_bottleneck._quantized_cdf)


@pytest.mark.parametrize("architecture", ["legacy", "plain4"])
@pytest.mark.parametrize("paths", [1, 2])
def test_initializer_and_calibration_cli_artifacts(tmp_path, monkeypatch, architecture, paths):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    initializer = __import__("initialize_hyper1d_from_vanilla")
    calibration = __import__("calibrate_hyper1d")
    monkeypatch.setattr(initializer, "GlobalSplat", lambda **kwargs:
                        small_model(architecture=kwargs["feature_codec"]["architecture"],
                                    paths=kwargs["feature_codec"]["paths"]))
    vanilla = tmp_path / "vanilla.ckpt"
    source = small_model(False).state_dict()
    torch.save({"state_dict": source}, vanilla)
    initial = tmp_path / "initial.ckpt"
    initializer.main(["--vanilla", str(vanilla), "--output", str(initial), "--architecture", architecture,
                      "--paths", str(paths)])
    artifact = torch.load(initial, weights_only=False)
    assert len(artifact["hyper1d_provenance"]["vanilla_checkpoint_sha256"]) == 64
    assert artifact["feature_codec_config"]["input_norm"] == "none"
    assert artifact["feature_codec_config"]["architecture"] == architecture
    assert artifact["feature_codec_config"]["paths"] == paths
    wrong_source = dict(source, scene_tokens=source["scene_tokens"][:1])
    wrong_path = tmp_path / "wrong.ckpt"
    torch.save({"state_dict": wrong_source}, wrong_path)
    with pytest.raises(ValueError, match="shapes"):
        initializer.main(["--vanilla", str(wrong_path), "--output", str(tmp_path / "bad.ckpt"), "--architecture", architecture])
    root = tmp_path / "re10k"
    (root / "train").mkdir(parents=True)
    (root / "train/index.json").write_text("{}")
    first, second = sample_batch(), sample_batch()
    second["scene_info"]["scene"] = ["another_scene"]
    monkeypatch.setattr(calibration, "build_model", lambda cfg: small_model(architecture=architecture, paths=paths))
    monkeypatch.setattr(calibration, "build_datamodule", lambda cfg:
        (SimpleNamespace(train_dataloader=lambda: [first, first, second]), None))
    calibrated = tmp_path / "calibrated.ckpt"
    calibration.main(["--checkpoint", str(initial), "--dataset-root", str(root),
                      "--output", str(calibrated), "--scenes", "2", "--device", "cpu"])
    artifact = torch.load(calibrated, weights_only=False)
    assert artifact["feature_codec_config"]["input_norm"] == "calibrated"
    manifest = artifact["hyper1d_provenance"]["calibration"]
    assert manifest["split"] == "train" and manifest["scene_count"] == 2
    assert manifest["token_count"] == 66 and len(manifest["geometry_projection_sha256"]) == 64
    assert "optimizer_states" not in artifact
    assert artifact["state_dict"]["model.feature_codec.f_std"].min() > 0


def install_mock_metrics(module, monkeypatch):
    import globalsplat.model.model_wrapper as wrapper

    class Benchmarker:
        def __init__(self):
            self.execution_times = defaultdict(list)

    psnr = lambda gt, pred: -10 * torch.log10((gt - pred).square().flatten(1).mean(1).clamp_min(1e-8))
    lpips = lambda gt, pred: (gt - pred).abs().flatten(1).mean(1)
    module._load_eval_utils = lambda: (psnr, lpips, lpips, Benchmarker, None, None)
    monkeypatch.setattr(wrapper, "render_static_batched", lambda preds, trg, **kw: {"img": torch.zeros_like(trg["images"]).flatten(0, 1)})
    module.log = lambda *args, **kwargs: None


@pytest.mark.parametrize("architecture", ["legacy", "plain4"])
@pytest.mark.parametrize("paths", [1, 2])
def test_actual_test_and_validation_byte_reports(tmp_path, monkeypatch, architecture, paths):
    module = GlobalSplatModule(small_model(architecture=architecture, paths=paths, use_morton=paths == 2).eval(), eval_mode=True, final_stage=0,
        test_cfg={"actual_bitstream": True, "output_path": str(tmp_path / "test")},
        validation_cfg={"actual_bitstream": True, "output_path": str(tmp_path / "val")},
        experiment_name="hyper1d_cpu")
    install_mock_metrics(module, monkeypatch)
    module._trainer = SimpleNamespace(global_step=2000, is_global_zero=True)
    batch = sample_batch()
    module.on_validation_start()
    module.validation_step(batch, 0)
    module.on_validation_epoch_end()
    row = json.loads((tmp_path / "val/step000002000.json").read_text())[0]
    streams = ["y", "z"] if paths == 1 else ["base_y", "base_z", "residual_y", "residual_z"]
    assert all(row[f"actual_{name}_bytes"] > 0 for name in streams)
    assert row["actual_container_bytes"] == (112 if paths == 1 else 172) and "entropy_bit_gap" in row
    assert row["actual_bytes"] == row["actual_container_bytes"] + sum(row[f"actual_{name}_bytes"] for name in streams)
    module.on_validation_start()
    module.validation_step(batch, 0)
    module.on_validation_epoch_end()  # identical manifest accepted
    module.on_test_start()
    module.test_step(batch, 0)
    module.on_test_end()
    report = json.loads((tmp_path / "test/hyper1d_cpu/scores_all_avg.json").read_text())
    assert all(report[f"actual_{name}_bytes"] > 0 for name in streams)
    assert report["actual_container_bytes"] == (112 if paths == 1 else 172)
    module.test_cfg["score_context_diagnostics"] = True
    with pytest.raises(ValueError, match="score-context"):
        module.on_test_start()


def test_pilot_config_and_runner_dry_run(tmp_path, capsys):
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(config_name="main", overrides=["+experiment=re10k_hyper1d_12h"])
        baseline = compose(config_name="main", overrides=["+experiment=re10k_32k_nfcgs"])
    assert cfg.trainer.max_steps == 16000 and cfg.trainer.max_time == "00:11:00:00"
    assert cfg.trainer.val_check_interval == 2000 * cfg.trainer.accumulate_grad_batches
    assert cfg.optimizer.min_lr_ratio == 1 and cfg.optimizer.warmup_pct * cfg.trainer.max_steps == 1000
    assert baseline.model.feature_codec.rank == 56 and "validation" not in baseline
    assert cfg.model.feature_codec.architecture == "plain4"
    assert cfg.model.feature_codec.n == 256 and cfg.model.feature_codec.m == 512
    output = tmp_path / "not_created"
    runner.main(["train", "--vanilla-checkpoint", "missing.ckpt", "--output", str(output), "--dry-run"])
    text = capsys.readouterr().out
    assert "initialize_hyper1d_from_vanilla.py" in text and "trainer.max_time=00:11:00:00" in text
    assert "--architecture plain4" in text and "model.feature_codec.architecture=plain4" in text
    assert not output.exists()
    args = runner.parse_args(["train", "--checkpoint", "codec.ckpt", "--accumulate", "2"])
    command = runner.build_command(args, output, Path("codec.ckpt"))
    assert "trainer.val_check_interval=4000" in command


def test_runner_legacy_initialization_and_checkpoint_architecture(tmp_path, capsys):
    from globalsplat.compression import FeatureHyperprior1DCodec, Hyper1DConfig
    runner.main(["train", "--vanilla-checkpoint", "missing.ckpt", "--architecture", "legacy", "--dry-run"])
    text = capsys.readouterr().out
    assert "--architecture legacy" in text and "model.feature_codec.n=192" in text
    assert "model.feature_codec.m=320" in text
    legacy = FeatureHyperprior1DCodec(Hyper1DConfig(texture_channels=4, geometry_channels=4,
        geometry_observable_channels=2, n=3, m=6, adapter_hidden=3))
    metadata = legacy.config.to_dict()
    del metadata["architecture"]
    checkpoint = tmp_path / "old.ckpt"
    torch.save({"state_dict": legacy.state_dict(), "feature_codec_config": metadata}, checkpoint)
    runner.main(["eval", "--checkpoint", str(checkpoint), "--dry-run"])
    assert "model.feature_codec.architecture=legacy" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        runner.parse_args(["train", "--checkpoint", str(checkpoint), "--resume", "--architecture", "plain4"])


def test_warmup_then_constant_scheduler():
    model = torch.nn.Linear(2, 2)
    setup = build_optimizer_and_scheduler(model, lr=1e-4, weight_decay=0,
        total_steps=16000, warmup_pct=0.0625, min_lr_ratio=1, optimizer_name="adam")
    optimizer, scheduler = setup["optimizer"], setup["lr_scheduler"]["scheduler"]
    for step in range(16000):
        optimizer.step()
        scheduler.step()
        if step in (999, 1999, 15999):
            assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_fixed_validation_uses_one_deterministic_stream(monkeypatch):
    @dataclass
    class DatasetConfig:
        view_sampler: object = None
        augment: bool = True
        shuffle_val: bool = True

    @dataclass
    class SamplerConfig:
        name: str
        num_context_views: int
        num_target_views: int

    class Scenes(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield from ({"scene": str(i)} for i in range(4))

    calls = []
    def get_dataset(config, stage, tracker):
        calls.append((config, stage))
        return Scenes()

    import importlib
    original_import = importlib.import_module
    def fake_import(name, *args, **kwargs):
        if name == "src.dataset.view_sampler.view_sampler_deterministic_all":
            return SimpleNamespace(ViewSamplerDeterministicAllCfg=SamplerConfig)
        if name == "src.dataset":
            return SimpleNamespace(get_dataset=get_dataset)
        if name == "src.dataset.data_module":
            return SimpleNamespace(worker_init_fn=None)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", fake_import)
    settings = SimpleNamespace(batch_size=2, num_workers=4, seed=123)
    upstream = SimpleNamespace(dataset_cfg=DatasetConfig(),
        data_loader_cfg=SimpleNamespace(val=settings, train=settings),
        step_tracker=None, dataset_shim=lambda dataset, stage: dataset,
        get_generator=lambda config: None, get_persistent=lambda config: True)
    dm = UpstreamBackedDataModule(upstream, None)
    monkeypatch.setattr(dm, "_collate_fn", lambda: torch.utils.data.default_collate)
    configure_fixed_validation(dm, {"context_views": 12, "target_views": 8})
    assert upstream.dataset_cfg.augment and upstream.dataset_cfg.shuffle_val
    assert not dm.fixed_validation_cfg.augment and not dm.fixed_validation_cfg.shuffle_val
    first, second = dm._make_loader("val"), dm._make_loader("val")
    assert first.batch_size == 1 and first.num_workers == 0
    assert list(first) == list(second) == [{"scene": [str(i)]} for i in range(4)]
    assert all(stage == "test" for config, stage in calls)


def test_timer_stop_final_snapshot_and_optimizer_resume(tmp_path):
    class Toy(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)
            self.validated_at = []

        def training_step(self, batch, batch_idx):
            return self.linear(batch).square().mean()

        def validation_step(self, batch, batch_idx):
            self.validated_at.append(self.global_step)

        def configure_optimizers(self):
            return build_optimizer_and_scheduler(self, lr=1e-4, weight_decay=0,
                total_steps=8, warmup_pct=0.25, min_lr_ratio=1)

    timer = InvocationTimer(duration="00:01:00:00", interval="step", verbose=False)
    timer.load_state_dict({"time_elapsed": {"train": 100000}})
    assert timer.time_elapsed() == 0
    loader = torch.utils.data.DataLoader(torch.randn(48, 2), batch_size=2)
    first = Toy()
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_model_summary=False, enable_progress_bar=False, callbacks=[timer],
        max_steps=8, accumulate_grad_batches=4, val_check_interval=8,
        check_val_every_n_epoch=None, limit_val_batches=1, num_sanity_val_steps=0)
    timer.time_elapsed = lambda *args: 4000 if trainer.global_step >= 2 else 0
    trainer.fit(first, loader, loader)
    path = tmp_path / "last.ckpt"
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["optimizer_states"] and checkpoint["lr_schedulers"]
    start = trainer.global_step
    assert start == 2 and first.validated_at == [2]
    second = Toy()
    resumed = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
        enable_model_summary=False, enable_progress_bar=False, max_steps=start + 2,
        callbacks=[InvocationTimer(duration="00:01:00:00", verbose=False)],
        accumulate_grad_batches=4, val_check_interval=8, check_val_every_n_epoch=None,
        limit_val_batches=1, num_sanity_val_steps=0)
    resumed.fit(second, loader, loader, ckpt_path=path)
    assert resumed.global_step == start + 2
    assert second.validated_at
