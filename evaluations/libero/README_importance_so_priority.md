# LIBERO-plus importance_so priority pipeline

This pipeline tests the current main method only.  It does not add more
FO/SO or importance/delta training arms.

All Query evaluations use residual strength 0.25.  Inner Mask adaptation keeps
`inner_lr=100` and scale 1.0.  Six paired variants are evaluated:

1. original seed7 checkpoint, normal and no-inner;
2. base-success-preserved checkpoint, normal and no-inner;
3. preserved checkpoint with the conservative gate, normal and no-inner.

The conservative gate accepts the meta action only when all enabled checks pass:

- the adapted Support loss does not increase;
- relative Query residual RMS is at most 0.10;
- normalized base/meta mean absolute action difference over the first five
  action steps and seven control dimensions is at most 0.05.

All thresholds are environment-variable overrides.  Meta and base action
chunks use identical flow noise before the gate chooses one complete chunk.
