"""Small codec configurations shared by the codec tests."""
from globalsplat.compression import CodecConfig

# The pre-cleanup CodecConfig layout defaults these tests were written against.
LEGACY_LAYOUT = dict(token_order="morton", score_context_schedule="legacy",
                     score_spatial_stages=2, score_spatial_kernel=3)


def config(**kwargs):
    return CodecConfig(**{
        **dict(texture_channels=8, geometry_channels=8, geometry_observable_channels=4,
               rank=7, residual_n=6, residual_m=8, adapter_hidden=6, score_slice_channels=3),
        **LEGACY_LAYOUT, **kwargs,
    })
