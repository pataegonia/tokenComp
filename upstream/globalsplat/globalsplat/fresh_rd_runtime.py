"""Opt-in Lightning hooks for the fresh RD experiments."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def checkpoint_protocol_matches(checkpoint, protocol):
    """Accept the exact protocol, or the explicitly recorded 60k repair parent."""
    saved = checkpoint.get("fresh_rd_protocol")
    if saved == protocol:
        return True
    parent = protocol.get("resume_parent", {}) if isinstance(protocol, dict) else {}
    if (not isinstance(saved, dict) or checkpoint.get("global_step") != 60000
            or saved != parent.get("protocol")):
        return False
    original = {k:v for k,v in saved.items() if k != "code_sha256"}
    current = {k:v for k,v in protocol.items() if k not in ("code_sha256", "resume_parent")}
    changed = {k for k in set(saved.get("code_sha256", {})) | set(protocol.get("code_sha256", {}))
               if saved.get("code_sha256", {}).get(k) != protocol.get("code_sha256", {}).get(k)}
    allowed = {"globalsplat/fresh_rd_runtime.py", "globalsplat/model/model_wrapper.py",
               "scripts/run_nfcgs_fresh_rd.py"}
    return (original == current and changed <= allowed
            and sorted(changed) == parent.get("changed_runtime_files")
            and parent.get("source_60k_mode") in ("stage_a_original", "resaved_after_first_microbatch"))


def override_scheduler_milestones(checkpoint, requested):
    """Edit the serialized MultiStepLR, before Lightning restores it."""
    requested = tuple(map(int, requested))
    states = checkpoint.get("lr_schedulers", [])
    if len(states) != 1 or "milestones" not in states[0] or not requested:
        raise ValueError("milestone override requires one saved MultiStepLR")
    state = states[0]
    step = int(checkpoint["global_step"])
    old = sorted(map(int, state["milestones"]))
    if step >= min(requested) or (old and step >= min(old)) or state["last_epoch"] >= min(requested):
        raise ValueError("milestones may change only before any LR decay")
    optimizers = checkpoint.get("optimizer_states", [])
    if len(optimizers) != 1:
        raise ValueError("one saved optimizer required")
    groups, bases = optimizers[0]["param_groups"], state["base_lrs"]
    if len(groups) != len(bases) or any(abs(g["lr"] - b) > 1e-12 for g, b in zip(groups, bases)):
        raise ValueError("refusing to raise an already-decayed optimizer LR")
    state["milestones"] = Counter(requested)
    return dict(step=step, old=old, new=list(requested), optimizer_lr_unchanged=True)


def stochastic_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_stochastic_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA RNG topology differs")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


class FreshRDState(pl.Callback):
    """Restore model-side randomness after the loader is re-created."""
    def __init__(self, trace=None):
        self.pending = None
        self.trace = Path(trace) if trace else None
        self.record = None

    def state_dict(self):
        return {"rng": stochastic_state()}

    def load_state_dict(self, state):
        self.pending = state["rng"]

    def on_train_start(self, trainer, pl_module):
        if self.pending is not None:
            restore_stochastic_state(self.pending)
            self.pending = None

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Lightning 2.3 saves inside on_train_batch_end, before incrementing
        # completed. A stateful loader has already consumed this batch. Leaving
        # completed one behind makes accumulation step on the first resumed
        # microbatch (batch_idx is then off by one). Normalize the serialized
        # progress only, after an optimizer boundary; the live loop is untouched.
        fit = checkpoint.get("loops", {}).get("fit_loop", {})
        progress = fit.get("epoch_loop.batch_progress", {})
        for key in ("total", "current"):
            counter = progress.get(key, {})
            if counter.get("ready") == counter.get("processed") and counter.get("completed", 0) < counter.get("processed", 0):
                counter["completed"] = counter["processed"]
        if "epoch_loop.state_dict" in fit:
            fit["epoch_loop.state_dict"]["_batches_that_stepped"] = int(trainer.global_step)

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if not self.trace:
            return
        self.record = dict(step=int(trainer.global_step), batch_idx=batch_idx)
        if isinstance(batch, dict) and "inputs" in batch:
            self.record.update(scene=batch["scene_info"]["scene"],
                context=batch["inputs"]["frame_ids"].detach().cpu().tolist(),
                target=batch["targets"]["frame_ids"].detach().cpu().tolist())
            if getattr(module, "subset_consistency", False):
                from .model.model_wrapper import make_anchor_alternating_input_subsets_and_shared_targets
                a, b, t, valid = make_anchor_alternating_input_subsets_and_shared_targets(
                    inputs=batch["inputs"], targets=batch["targets"],
                    min_shared_targets=module.subset_min_targets,
                    flip=(int(module.global_step) % 2 == 1) if module.subset_flip_parity else False)
                self.record.update(subset_a=a["frame_ids"].detach().cpu().tolist(),
                    subset_b=b["frame_ids"].detach().cpu().tolist(),
                    shared_target=t["frame_ids"].detach().cpu().tolist(), valid=valid.detach().cpu().tolist())
        else:
            self.record["sample"] = batch[0].detach().cpu().tolist()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.trace:
            row = dict(self.record, end_step=int(trainer.global_step),
                loss=float(outputs["loss"].detach().cpu()) if isinstance(outputs, dict) else float(outputs),
                lr=[g["lr"] for g in trainer.optimizers[0].param_groups])
            self.trace.parent.mkdir(parents=True, exist_ok=True)
            with self.trace.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")


class FreshRDCheckpoint(ModelCheckpoint):
    """Save after the module's quantile update, including at stage boundaries."""
    def on_train_start(self, trainer, pl_module):
        super().on_train_start(trainer, pl_module)
        # Lightning 2.3 does not serialize this callback field. At an accumulation
        # boundary the first resumed microbatch still has the saved global_step.
        self._last_global_step_saved = int(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.on_train_batch_end(outputs, batch, batch_idx)
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
