# QUESTS: Quick Uncertainty Estimation via STructural Similarity

QUESTS provides model-free uncertainty estimation methods for interatomic potentials.
Among the methods, we propose a structural descriptor based on k-nearest neighbors that is:

1. Fast to compute, as it uses only distances between atoms within an environment.
2. Invertible, as the distances can be used to reconstruct an environment up to an invariant.
3. Interpretable, as the distances correspond to directly to displacements of atoms.

This package also contains metrics to quantify the diversity of a dataset using this descriptor, and tools to interface with other representations and packages.

## Installation

To use the `quests` package, clone this repository from GitLab and use `pip` to install it into your virtual environment:

```bash
git clone ssh://git@czgitlab.llnl.gov:7999/dskoda/quests.git
cd quests
pip install .
```

## Usage

TO-DO: For now, please contact Daniel (schwalbekoda1) for help with the package
