# QUESTS: Quick Uncertainty and Entropy from STructural Similarity

QUESTS provides model-free uncertainty and entropy estimation methods for interatomic potentials.
Among the methods, we propose a structural descriptor based on k-nearest neighbors that is:

1. Fast to compute, as it uses only distances between atoms within an environment.
Because the descriptors are parallelized, generation of descriptors for 1.5M environments takes about 3 seconds on a 56-core computer.
2. Interpretable, as the distances correspond to directly to displacements of atoms.
3. Lead to an interpretable information entropy value.

This package also contains metrics to quantify the diversity of a dataset using this descriptor, and tools to interface with other representations and packages.

## Installation

### From repository

To install the `quests` package directly from the repository, clone it from GitHub and use `pip` to install it into your virtual environment:

```bash
git clone https://github.com/dskoda/quests.git
cd quests
pip install .
```

## Usage

### Command line

Once installed, you can use the `quests` command to perform different analyses. For example, to compute the entropy of any dataset (the input can be anything that ASE reads, including xyz files), you can use the `quests entropy` command:

```bash
quests entropy dump.lammpstrj --bandwidth 0.015
```

For subsampling the dataset and avoiding using the entire dataset, use the `entropy_sampler` example:

```bash
quests entropy_sampler dataset.xyz --batch_size 20000 -s 100000 -n 3
```

`-s` specifies the number of sampled environments, `-n` specifies how many runs will be computed (for statistics).

For additional help with these commands, please use `quests --help`, `quests entropy --help`, and others.

### API

#### Computing descriptors and dataset entropy

To use the QUESTS package to create descriptors and compute entropies, you can use the [descriptor](quests/descriptor.py) and [entropy](quests/entropy.py) submodules:

```python
from ase.io import read
from quests.descriptor import get_descriptors
from quests.entropy import perfect_entropy, diversity

dset = read("dataset.xyz", index=":")
x = get_descriptors(dset, k=32, cutoff=5.0)
h = 0.015
batch_size = 10000
H = perfect_entropy(x, h=h, batch_size=batch_size)
D = diversity(x, h=h, batch_size=batch_size)
```

In this example, descriptors are being created using 32 nearest neighbors and a 5.0 Å cutoff.
The entropy and diversity are being computed using a Gaussian kernel (default) with bandwidth of 0.015 1/Å and batch size of 10,000.

#### Computing differential entropies

```python
from ase.io import read
from quests.descriptor import get_descriptors
from quests.entropy import delta_entropy

dset_x = read("reference.xyz", index=":")
dset_y = read("test.xyz", index=":")

k, cutoff = 32, 5.0
x = get_descriptors(dset_x, k=k, cutoff=cutoff)
y = get_descriptors(dset_y, k=k, cutoff=cutoff)

# computes dH (Y | X)
dH = delta_entropy(y, x, h=0.015)
```

#### Computing approximate differential entropies

```python
from ase.io import read
from quests.descriptor import get_descriptors
from quests.entropy import approx_delta_entropy

dset_x = read("reference.xyz", index=":")
dset_y = read("test.xyz", index=":")

k, cutoff = 32, 5.0
x = get_descriptors(dset_x, k=k, cutoff=cutoff)
y = get_descriptors(dset_y, k=k, cutoff=cutoff)

# approximates dH (Y | X)
# n = 5 and graph_neighbors = 10 are arguments for
# pynndescent, which performs an approximate nearest
# neighbor search for dH
dH = approx_delta_entropy(y, x, h=0.015, n=5, graph_neighbors=10)
```

### Citing

If you use QUESTS in a publication, please cite the following paper:

```bibtex
@article{schwalbekoda2024information,
    title = {Information theory unifies atomistic machine learning, uncertainty quantification, and materials thermodynamics},
    author = {Schwalbe-Koda, Daniel and Hamel, Sebastien and Sadigh, Babak and Zhou, Fei and Lordi, Vincenzo},
    year = {2024},
    journal = {arXiv:},
    doi = {},
    url = {},
    arxiv={},
}
```
## License

The QUESTS software is distributed under the following license: BSD-3

All new contributions must be made under this license.

SPDX: BSD-3-Clause

## Acknowledgements

This work was produced under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344, with support from LLNL's LDRD program under tracking codes 22-ERD-055 and 23-SI-006.

Code released as LLNL-CODE-858914
