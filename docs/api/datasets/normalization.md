# Normalization

Flow matching transports $\mathcal{N}(0, I)$ to the data, so raw PDE fields with
non-unit scale make the velocity regression badly conditioned. Always normalize.

Statistics are keyed by **raw field name** (`source`, `solution`, `kappa`, `initial`,
`final`) rather than by role, which is why one normalizer stays correct across
`problem='forward'` and `problem='inverse'`.

!!! warning
    Never refit normalization on validation or test data — share the *same* train
    normalizer instance.

::: flowpde.datasets.normalization
