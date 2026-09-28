# NFC-GS Full cooldown and score-probability experiment (2026-09-13)

## Decision

The completed P0/P1/P2 runs are not rerun. Their `additional25k` stage reset Adam
at `1e-4`, while the 50k parent had already reached `1e-6`; the resulting parent
drift is larger than the predictor differences. The follow-up therefore separates
two questions:

1. Can a reconstruction-changing Full model improve when continued at the
   parent's final learning rate?
2. Can score bytes be reduced without changing the decoded representation at all?

All comparisons use the original 50k Full parents (nonlinear task 8/9), the same
seed, batch, view protocol, and schedule at both lambdas.

## Implemented probability models

`model.feature_codec.score_spatial_entropy` accepts:

- `shared`: existing factorized entropy model for both even and odd symbols;
- `split`: independent factorized entropy models for even and odd symbols;
- `gaussian`: learned per-channel static Gaussian scale for odd symbols;
- `conditional_scale`: odd-symbol Gaussian scale predicted causally from already
  decoded even-anchor residuals.

The conditional model never reads undecoded odd symbols. All variants retain the
existing even/odd stream layout, so they add neither a stream nor a decode pass.
The bitstream flags distinguish predictor and entropy variants and reject a
mismatched decoder configuration.

`feature_codec_train_scope=score_probability` freezes the basis, transforms,
score offsets/steps, context value predictors, residual codec, and entropy
quantiles. It trains only probability-density parameters and optional Gaussian  
scales. Consequently its decoded features and PSNR are invariant; only estimated
and actual rate can change. The Slurm recipe also sets
`loss.quantile_update_interval=0` to enforce the same invariant operationally.

Raw XYZ conditioning was deliberately not added: XYZ is not currently available
to the score entropy decoder from the portable bitstream, so using it would either
be circular or require a new side stream.

## Grid A: fair Full low-LR continuation

Every arm starts from the same original Full checkpoint for its lambda. P2 adds a
zero-initialized residual-7 predictor to that parent; P0 is the unchanged linear
predictor. Both train all codec parameters for 10k steps at constant `1e-6`.

| Task | Lambda | Predictor | Parent |
|---:|---:|---|---:|
| 0 | 0.0064 | P0 linear | nonlinear task 8, step 50k |
| 1 | 0.0064 | P2 residual7 | nonlinear task 8, step 50k |
| 2 | 0.0256 | P0 linear | nonlinear task 9, step 50k |
| 3 | 0.0256 | P2 residual7 | nonlinear task 9, step 50k |

```bash
bash scripts/slurm/train_nfcgs_full_cooldown.slurm plan
bash scripts/slurm/submit_nfcgs_full_cooldown.sh dry-run
bash scripts/slurm/submit_nfcgs_full_cooldown.sh submit
```

Training output:
`outputs/nfcgs_full_lowlr10k/<run>/<arm>/.../step000010000.ckpt`

Evaluation output:
`outputs/nfcgs_full_lowlr10k_eval/<run>/<arm>/...`

## Grid B: reconstruction-locked score probability fit

Every arm starts from the original Full P0 parent and trains for 10k steps with
`score_probability`, Adam `1e-4`, milestone 7k, gamma 0.1, and no quantile update.
The shared arm is the control for continuation effects.

| Task | Lambda | Entropy model | Tag |
|---:|---:|---|---|
| 0 | 0.0064 | shared factorized | e0_shared |
| 1 | 0.0256 | shared factorized | e0_shared |
| 2 | 0.0064 | even/odd split factorized | e1_split |
| 3 | 0.0256 | even/odd split factorized | e1_split |
| 4 | 0.0064 | static per-channel Gaussian | e2_gaussian |
| 5 | 0.0256 | static per-channel Gaussian | e2_gaussian |
| 6 | 0.0064 | causal conditional Gaussian scale | e3_conditional_scale |
| 7 | 0.0256 | causal conditional Gaussian scale | e3_conditional_scale |

```bash
bash scripts/slurm/train_nfcgs_score_probability.slurm plan
bash scripts/slurm/submit_nfcgs_score_probability.sh dry-run
bash scripts/slurm/submit_nfcgs_score_probability.sh submit
```

Training output:
`outputs/nfcgs_score_probability10k/<run>/<entropy-tag>/.../step000010000.ckpt`

Evaluation output:
`outputs/nfcgs_score_probability10k_eval/<run>/<entropy-tag>/...`

## Result interpretation

- Compare actual score bytes within each lambda; PSNR should be identical for all
  probability-only arms. A PSNR change indicates an invariant violation.
- `split < shared` shows a useful even/odd marginal difference without needing a
  spatially conditional model.
- `conditional_scale < gaussian` isolates the value of decoder-causal local
  context beyond a static per-channel scale.
- Promote the best probability model only if total KiB improves, not merely the
  score component, and round-trip decoding remains exact.
- Compare cooldown P0/P2 against their common 50k parent and against each other.
  If neither beats the parent, stop spending compute on score value-predictor
  depth and keep the original Full checkpoint as the reconstruction model.

## Local verification

- score entropy round-trip, bitstream guard, causal scale, trainable scope,
  warm-start/reconstruction invariance, and checkpoint strict-load: 43 tests passed;
- new Slurm mappings, exact checkpoint paths, optimizer overrides, and dry-run:
  25 tests passed;
- complete test suite: 248 passed; one pre-existing unrelated failure remains in
  `test_default_eval_is_not_pinned_to_one_node` because
  `eval_nfcgs_transform.slurm` is already pinned to `ariel-v12`.
