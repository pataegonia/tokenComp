"""GlobalSplat: the feed-forward image(s) -> 3D Gaussians model.

This is the whole model in one place. It used to be a three-deep wrapper chain
(GlobalSplat -> ImageRayFusion -> SceneTokenFusion); those thin layers are
collapsed here into a single module with three clear stages:

    1. tokenize   each view into RGB-patch tokens + camera/ray tokens,
    2. encode     a fixed bank of learnable scene tokens by cross-attending the
                  view tokens (the dual-stream slot encoder),
    3. decode     the refined scene tokens into an explicit set of 3D Gaussians.

``forward`` returns a typed ``Gaussians`` (see ``model/types.py``) that the
gsplat renderer consumes directly. The rasterizer itself lives in
``model/rendering.py`` and is driven by the LightningModule.
"""
from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .types import Gaussians
from .encoder.dual_stream import DualStreamSlotEncoder
from .decoder.gaussian_decoder import TokenCoarseToFine3DGS
from ..misc.pose_encoding import PatchViewPE, compute_rays


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class GlobalSplat(nn.Module):
    def __init__(
        self,
        sh_degree: int = 3,
        static_only: bool = False,
        use_camera_diff_as_input: bool = False,
        patch_size: int | Tuple[int, int] = 8,
        latent_rep_token_amount: int = 4096,
        dim_latents: int = 512,
        dim_rays: int = 256,
        dim_rgb_feat: int = 512,
        rounds: int = 4,
        slot_calib_layers_per_round: int = 2,
        num_heads: int = 8,
        M_max: int = 16,
        feature_codec: Mapping[str, Any] | None = None,
        freeze_globalsplat: bool = False,
        feature_codec_train_scope: str = "all",
        score_mean_offset_enabled: bool = True,
    ):
        super().__init__()

        if not static_only:
            raise NotImplementedError("Dynamic/motion decoding is not implemented yet.")

        self.sh_degree = sh_degree
        self.static_only = static_only
        self.use_camera_diff_as_input = use_camera_diff_as_input
        self.patch_size = self._normalize_patch_size(patch_size)
        self.dim_rays = dim_rays
        self.dim_rgb_feat = dim_rgb_feat
        self.dim_latents = dim_latents

        ph, pw = self.patch_size

        # ----- (1) tokenization -----
        # RGB patches -> tokens.
        self.rgb_patch_embeds = nn.Sequential(
            Rearrange("b v c (hh ph) (ww pw) -> b v (hh ww) (ph pw c)", ph=ph, pw=pw),
            nn.Linear(3 * (ph ** 2), self.dim_rgb_feat, bias=False),
        )
        # Plucker-ray patches -> tokens.
        self.ray_patch_embeds = nn.Sequential(
            Rearrange("b v c (hh ph) (ww pw) -> b v (hh ww) (ph pw c)", ph=ph, pw=pw),
            nn.Linear(6 * (ph ** 2), self.dim_rays, bias=False),
        )
        # Per-view camera conditioning added to the ray tokens.
        self.view_pe = PatchViewPE(token_dim=self.dim_rays)

        self.register_buffer(
            "_resnet_mean",
            torch.tensor(_RESNET_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_resnet_std",
            torch.tensor(_RESNET_STD, dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        content_dim = self.dim_rgb_feat + self.dim_rays

        # ----- (2) learnable scene-token bank + dual-stream slot encoder -----
        n_tokens = int(latent_rep_token_amount)
        init_std = 0.02
        self.scene_tokens = nn.Parameter(init_std * torch.randn(n_tokens, self.dim_latents))
        if n_tokens <= self.dim_latents:
            # Small bank: initialize the rows to a (scaled) orthogonal set.
            with torch.no_grad():
                q, _ = torch.linalg.qr(torch.randn(self.dim_latents, self.dim_latents))
                self.scene_tokens.copy_(init_std * q[:n_tokens].contiguous())

        self.slot_encoder = DualStreamSlotEncoder(
            dim_latent=self.dim_latents,
            dim_token=content_dim,
            heads=num_heads,
            use_camera_diff_as_input=self.use_camera_diff_as_input,
            rounds=rounds,
            slot_calib_layers_per_round=slot_calib_layers_per_round,
        )

        # Re-initialize the encoder's Linear/LayerNorm weights (matches the
        # original SceneTokenFusion init scope: encoder only -- the tokenization
        # embeds keep their default init and the decoder is initialized below by
        # its own constructor, after this pass).
        self._init_encoder_weights()

        # Optional NFC-GS boundary: appearance stays 512-D while geometry is
        # projected to the decoder-observable subspace before entropy coding.
        self.feature_codec = None
        geometry_decoder_dim = self.dim_latents
        if feature_codec is not None:
            from ..compression import CodecConfig, ObservableLowRank1DCodec

            codec_values = dict(feature_codec)
            codec_values.setdefault("texture_channels", self.dim_latents)
            codec_values.setdefault("geometry_channels", self.dim_latents)
            codec_config = CodecConfig.from_mapping(codec_values)
            if codec_config.texture_channels != self.dim_latents:
                raise ValueError("feature_codec.texture_channels must match dim_latents")
            if codec_config.geometry_channels != self.dim_latents:
                raise ValueError("feature_codec.geometry_channels must match dim_latents")
            self.feature_codec = ObservableLowRank1DCodec(codec_config)
            geometry_decoder_dim = codec_config.geometry_observable_channels

        # Set after each codec-enabled forward so training/evaluation code can
        # consume likelihoods and estimated bits without changing Gaussians.
        self.last_codec_output = None

        # ----- (3) Gaussian decoder -----
        # M_max = candidate splats predicted per token before coarse-to-fine
        # gating (the curriculum's final_stage must satisfy 2**final_stage <=
        # M_max). The 16K/32K models use M_max=16; the 2K model uses M_max=2.
        self.gaussian_decoder = TokenCoarseToFine3DGS(
            C=self.dim_latents,
            geometry_C=geometry_decoder_dim,
            sh_degree=self.sh_degree,
            M_max=M_max,
        )
        self.freeze_globalsplat = bool(freeze_globalsplat)
        self.feature_codec_train_scope = str(feature_codec_train_scope)
        if self.feature_codec is not None and self.feature_codec.score_context is not None:
            self.feature_codec.score_context.mean_offset_enabled = bool(
                score_mean_offset_enabled
            )
        if self.freeze_globalsplat:
            self.freeze_globalsplat_backbone()

    def freeze_globalsplat_backbone(self) -> None:
        """Freeze released GlobalSplat weights while leaving the codec trainable."""

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if self.feature_codec is None:
            raise ValueError("freeze_globalsplat requires feature_codec")
        self.feature_codec.set_trainable_scope(self.feature_codec_train_scope)

    @staticmethod
    def _normalize_patch_size(patch_size: int | Tuple[int, int]) -> Tuple[int, int]:
        return (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)

    def _init_encoder_weights(self) -> None:
        with torch.no_grad():
            for module in self.slot_encoder.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    module.reset_parameters()

    def set_stage(self, stage: int, mix: float = 1.0) -> None:
        """Set the coarse-to-fine stage; cascades into the Gaussian decoder."""
        self.stage = int(stage)
        self.mix = float(mix)
        self.gaussian_decoder.set_stage(stage, mix=mix)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------
    def _tokenize(self, images: torch.Tensor, cam_int: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
        """Return per-patch content tokens [B, V, P, dim_rays + dim_rgb_feat]."""
        norm_images = (images - self._resnet_mean) / self._resnet_std

        image_tokens = self.rgb_patch_embeds(norm_images)

        _, _, _, height, width = images.shape
        rays = compute_rays(c2w, K=cam_int, h=height, w=width, device=images.device)
        ray_tokens = self.ray_patch_embeds(rays)
        ray_tokens = self.view_pe(tokens=ray_tokens, c2w=c2w, K=cam_int, images=norm_images)

        return torch.cat([ray_tokens, image_tokens], dim=-1)

    def encode_scene_tokens(self, inputs) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GlobalSplat through `_pack_outputs` and return (appearance, geometry)."""

        images = inputs["images"]
        cam_int = inputs["intrinsic"]
        c2w = inputs.get("c2w", None)
        content_tokens = self._tokenize(images, cam_int, c2w)
        batch_size = content_tokens.shape[0]
        scene_state = self.scene_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        _aux, encoded_patch, _global = self.slot_encoder(content_tokens, state=scene_state)
        return encoded_patch

    def _decode_scene_tokens(self, encoded_patch) -> Gaussians:
        means, rotations, scales, sh, opacities, reg = self.gaussian_decoder(encoded_patch)
        return Gaussians(
            means=means,
            scales=scales,
            rotations=rotations,
            sh=sh,
            opacities=opacities,
            reg=reg,
        )

    @torch.no_grad()
    def compress_scene_tokens(self, encoded_patch):
        """Entropy-code already encoded scene tokens."""
        if self.feature_codec is None:
            raise RuntimeError(
                "compress_scene_tokens requires a feature_codec configuration"
            )
        appearance, geometry = encoded_patch
        geometry_observable = self.feature_codec.project_geometry(geometry)
        token_centers = self.gaussian_decoder.decoded_token_centers(geometry_observable)
        return self.feature_codec.compress(appearance, geometry, token_centers)

    @torch.no_grad()
    def decompress_scene_tokens(self, data: bytes):
        """Entropy-decode an `E2EM0301` scene into decoder input tokens."""
        if self.feature_codec is None:
            raise RuntimeError(
                "decompress_scene_tokens requires a feature_codec configuration"
            )
        return self.feature_codec.decompress(data)

    def decode_scene_tokens(self, encoded_patch) -> Gaussians:
        """Convert reconstructed scene tokens into renderable Gaussians."""
        return self._decode_scene_tokens(encoded_patch)

    @torch.no_grad()
    def compress_scene(self, inputs):
        """Encode one image-conditioned scene into an `E2EM0301` bitstream."""

        return self.compress_scene_tokens(self.encode_scene_tokens(inputs))

    @torch.no_grad()
    def decompress_scene(self, data: bytes) -> Gaussians:
        """Decode an `E2EM0301` scene directly to a renderable Gaussian set."""

        return self.decode_scene_tokens(self.decompress_scene_tokens(data))

    def forward(self, inputs) -> Gaussians:
        """Forward pass.

        Args:
            inputs (dict): keys ``images`` [B, V, 3, H, W], ``intrinsic``
                (pixel-space), and ``c2w`` (camera-to-world). ``extrinsic`` may
                be present but is not used here.

        Returns:
            Gaussians: the predicted 3D Gaussian set plus the decoder regularizer.
        """
        encoded_patch = self.encode_scene_tokens(inputs)

        self.last_codec_output = None
        if self.feature_codec is not None:
            appearance, geometry = encoded_patch
            geometry_observable = self.feature_codec.project_geometry(geometry)
            token_centers = self.gaussian_decoder.decoded_token_centers(geometry_observable)
            self.last_codec_output = self.feature_codec(
                appearance,
                geometry,
                token_centers,
                # Gaussian decoding is token-wise and the output is a set, so
                # no inverse permutation needs to be transmitted or applied.
                restore_original_order=False,
                training=self.training,
            )
            encoded_patch = (
                self.last_codec_output.texture,
                self.last_codec_output.geometry_observable,
            )

        return self._decode_scene_tokens(encoded_patch)
