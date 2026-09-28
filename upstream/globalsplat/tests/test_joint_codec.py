"""CPU checks for the scratch encoder -> codec -> Gaussian decoder gradient path."""

import torch

from globalsplat.model.globalsplat import GlobalSplat


def test_scratch_gradients_reach_encoder_codec_and_decoder():
    torch.manual_seed(91)
    model = GlobalSplat(
        static_only=True,
        sh_degree=0,
        patch_size=4,
        latent_rep_token_amount=32,
        dim_latents=16,
        dim_rays=16,
        dim_rgb_feat=16,
        rounds=1,
        slot_calib_layers_per_round=1,
        num_heads=4,
        M_max=2,
        freeze_globalsplat=False,
        feature_codec=dict(
            geometry_observable_channels=8,
            rank=4,
            residual_n=6,
            residual_m=8,
            adapter_hidden=6,
            transform_hidden=8,
            score_slice_channels=2,
            score_context_hidden=8,
        ),
    ).train()
    model.set_stage(0)
    images = torch.rand(1, 2, 3, 8, 8)
    cameras = torch.eye(4).repeat(1, 2, 1, 1)
    cameras[:, 1, 0, 3] = 0.1
    intrinsics = torch.tensor(
        [[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]]
    ).repeat(1, 2, 1, 1)
    gaussians = model(dict(images=images, intrinsic=intrinsics, c2w=cameras))
    assert gaussians.num_gaussians == 32
    assert model.last_codec_output is not None
    # A differentiable Gaussian-space objective isolates the model graph from
    # the CUDA rasterizer; this is not a rendering/convergence test.
    distortion = gaussians.means.square().mean() + gaussians.sh.square().mean()
    distortion.backward(retain_graph=True)
    for module in (
        model.rgb_patch_embeds,
        model.slot_encoder,
        model.feature_codec,
        model.gaussian_decoder,
    ):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    model.zero_grad(set_to_none=True)
    (model.last_codec_output.estimated_bits / 32).backward()
    for module in (model.rgb_patch_embeds, model.feature_codec.score_context):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
