# Example Notebooks

Jupyter notebooks in [`notebooks/`](https://github.com/sarperyn/FlowPDE/tree/main/notebooks)
covering dataset exploration and training workflows.

## Dataset Exploration

These notebooks generate a dataset, report its statistics, and visualize samples —
the right starting point for understanding what the flow is being asked to learn.

<div class="grid cards" markdown>

-   :material-notebook: **[Poisson Dataset — EDA](https://github.com/sarperyn/FlowPDE/blob/main/notebooks/poisson_dataset.ipynb)**

    Source → solution pairs for $\nabla^2 u = f$: generation, statistics, and sample
    visualization.

-   :material-notebook: **[Poisson](https://github.com/sarperyn/FlowPDE/blob/main/notebooks/poisson.ipynb)**

    An end-to-end 2D Poisson workflow, from dataset generation to learned solutions.

-   :material-notebook: **[Burgers Dataset — EDA](https://github.com/sarperyn/FlowPDE/blob/main/notebooks/burgers_dataset.ipynb)**

    Initial-condition → final-state pairs, with trajectory rollouts.

</div>

## Training and Analysis

<div class="grid cards" markdown>

-   :material-notebook: **[Burgers](https://github.com/sarperyn/FlowPDE/blob/main/notebooks/burgers.ipynb)**

    Training a flow on the Burgers equation and inspecting the results.

-   :material-notebook: **[Darcy](https://github.com/sarperyn/FlowPDE/blob/main/notebooks/darcy.ipynb)**

    Variable-coefficient Poisson, $-\nabla \cdot (\kappa \nabla u) = f$ — the setting
    where the inverse problem has a genuinely non-degenerate posterior.

</div>

!!! note "On the Poisson source distribution"
    The Poisson dataset currently ships an "easy" variant — 3 sine terms at wavenumbers
    1–3 (`source_num_terms`, `source_max_mode`). That is a low-dimensional subspace and
    produces optimistic errors. Restore a harder source distribution before reporting
    operator-learning results.

## Running Locally

```bash
pip install -e ".[docs]"   # includes jupyter via mkdocs-jupyter
jupyter notebook notebooks/
```

Detailed ablations and benchmark configurations will be documented in the project
report. The notebooks remain the public, runnable examples for the library.
