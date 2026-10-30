# T06 frozen H+D4 order intervention

Matched scenes: 256; same F, projection, bases, normalized Score, XYZ and weights.
Hilbert packet SHA-256 matches completed T05 for every scene.

| Order | Total KiB | Score raw KiB | MSH y KiB | MSH z KiB | Header B | Anchor XYZ distance | Score residual normalized MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| slot | 56.67001 | 41.06366 | 15.19284 | 0.13617 | 284.0 | 2.3873300 | 1.6707784 |
| morton | 37.91431 | 23.28050 | 14.22141 | 0.13506 | 284.0 | 0.2421056 | 1.5297684 |
| hilbert | 37.25922 | 22.66045 | 14.18643 | 0.13499 | 284.0 | 0.2092603 | 1.5245647 |
| random10_mean | 56.72330 | 41.11120 | 15.19830 | 0.13645 | 284.0 | 2.3901985 | 1.6718214 |

Paired total bytes minus Hilbert (95% scene bootstrap CI):

- slot: +19876.656 B [+19598.986, +20176.601].
- morton: +670.812 B [+648.911, +696.355].
- random10_mean: +19931.219 B [+19648.911, +20233.803].

Frozen Hilbert+D4 predictor and model; non-Hilbert permutations are distribution-shift diagnostics, not retrained RD curves.
1000 paired scene resamples; random orders averaged within scene first; not training-seed uncertainty
Actual causal reconstructed anchors condition coding; observer statistics never condition the model.
Score raw excludes its wrapper. Headers include both Score wrapper and scene/MSH containers.
MSH residual is recomputed for each decoded Score; all permutations affect both Score and MSH.
No rendering, PSNR or full-test RD claim. Raw traces contain exact anchors/predictions for selected scenes/orders.
