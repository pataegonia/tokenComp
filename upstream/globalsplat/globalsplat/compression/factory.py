"""Dispatch codec construction without changing the legacy constructor."""

from typing import Any, Mapping
from .config import CodecConfig
from .codec import ObservableLowRank1DCodec
from .hyper1d_config import Hyper1DConfig
from .hyper1d import FeatureHyperprior1DCodec


def codec_config_from_mapping(values: Mapping[str, Any]):
    kind = values.get("codec_type")
    if kind is None:
        return CodecConfig.from_mapping(values)
    if kind == "hyper1d":
        return Hyper1DConfig.from_mapping(values)
    raise ValueError(f"unsupported codec_type: {kind!r}")


def build_feature_codec(config):
    if isinstance(config, Hyper1DConfig):
        return FeatureHyperprior1DCodec(config)
    if isinstance(config, CodecConfig):
        return ObservableLowRank1DCodec(config)
    raise TypeError(f"unsupported codec configuration: {type(config)}")
