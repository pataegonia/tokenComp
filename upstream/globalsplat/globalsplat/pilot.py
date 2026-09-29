"""Opt-in helpers for a timed codec pilot; baseline training is unaffected."""

from dataclasses import replace
import importlib
from pytorch_lightning.callbacks import Timer


class InvocationTimer(Timer):
    """Give each resumed job its own wall-clock allowance."""

    def load_state_dict(self, state_dict):
        # Optimizer/global step still resume; elapsed time from an earlier job does not.
        self._offset = 0


def configure_fixed_validation(datamodule, config):
    sampler_type = importlib.import_module(
        "src.dataset.view_sampler.view_sampler_deterministic_all"
    ).ViewSamplerDeterministicAllCfg
    sampler = sampler_type(name="deterministic_all",
                           num_context_views=int(config.get("context_views", 12)),
                           num_target_views=int(config.get("target_views", 8)))
    datamodule.fixed_validation_cfg = replace(datamodule._dm.dataset_cfg,
                                              view_sampler=sampler, augment=False, shuffle_val=False)
