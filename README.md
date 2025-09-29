# QUESTS: Quick Uncertainty and Entropy via STructural Similarity

[![Code](https://zenodo.org/badge/760951897.svg)](https://doi.org/10.5281/zenodo.15025957)

QUESTS provides model-free uncertainty and entropy estimation methods for interatomic potentials.
Among the methods, we propose a structural descriptor and information-theoretical strategy that:

1. Is fast to compute, as it uses only distances between atoms within an environment.
Because the computation of descriptors is efficiently parallelized, generation of descriptors for 1.5M environments takes about 3 seconds on 56 threads (tested against Intel Xeon CLX-8276L CPUs).
2. Can be used to analyze datasets for atomistic machine learning, providing quantities such as dataset entropy, diversity, information gap, and others.
3. Is shown to recover many useful properties of information theory, and can be used to inform dataset compression
4. Has useful properties in terms of outlier detection and even some thermodynamic correlations under study!

This package also contains tools to interface with other representations and packages, as described below.

## Installation

### From pip

```bash
pip install quests
```

### From repository

To install the `quests` package directly from the repository, clone it from GitHub and use `pip` to install it into your virtual environment:

```bash
git clone https://github.com/dskoda/quests.git
cd quests
pip install .
```

Typical installation times are on the order of a minute, and depend mostly on the internet connection to download the dependencies and on conflict resolution by the package manager (if applicable).

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
from quests.entropy import entropy, diversity

dset = read("dataset.xyz", index=":")
x = get_descriptors(dset, k=32, cutoff=5.0)
h = 0.015
batch_size = 10000
H = entropy(x, h=h, batch_size=batch_size)
D = diversity(x, h=h, batch_size=batch_size)
```

In this example, descriptors are being created using 32 nearest neighbors and a 5.0 Å cutoff.
The entropy and diversity are being computed using a Gaussian kernel (default) with bandwidth of 0.015 1/Å and batch size of 10,000.
For multicomponent systems, see below.

#### Computing descriptors for multicomponent systems

As of the current version, QUESTS allow you to compute a simplified descriptor for multi-element systems.
The descriptor is made by the concatenation of normal QUESTS descriptors, and the same descriptor computed on a per-element basis.
For instance, for a hypothetical A-B alloy, we first compute the descriptors on a per-environment basis without regard for composition.
Then, we compute it two more times: one considering that only atoms A exist, then later considering that only atoms B exist.
The final descriptor is made by the concatenation of these three descriptors.
For N elements, the final length of the descriptor is (N + 1) * k * (k + 1) / 2, where k is the number of neighbors passed as a parameter.

```python
from ase.io import read
from quests.descriptor import get_descriptors_multicomponent
from quests.entropy import perfect_entropy, diversity

dset = read("dataset.xyz", index=":")
species = ["Ag", "Au"]  # if not provided, species will be inferred from the dataset
x = get_descriptors_multicomponent(dset, k=32, cutoff=5.0, species=species)
```

Like in the example before, descriptors are being created using 32 nearest neighbors and a 5.0 Å cutoff.
Please note: as of now, the rule for the selection of the bandwidth is still being tested.
In principle, however, all the other functions for QUESTS are still compatible with the multicomponent descriptors.

⚠️ Note: This feature is under construction and is subject to change

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

#### Computing the dataset entropy using PyTorch

To accelerate the computation of entropy of datasets, one can use PyTorch to compute the entropy of a system.
This can be done by first installing the optional dependencies for this repository:

```bash
pip install quests[gpu]
```

The syntax of the entropy, as computed with PyTorch, is identical to the one above.
Instead of loading the functions from [quests.entropy](quests/entropy.py), however, you should load them from [quests.gpu.entropy](quests/gpu/entropy.py).
The descriptors remain the same - as of now, creating descriptors using GPUs is not supported.
Note that this constraint requires the descriptors to be generated using the traditional routes, and later converted into a `torch.tensor`.

```python
import torch
from ase.io import read
from quests.descriptor import get_descriptors
from quests.gpu.entropy import entropy

dset = read("dataset.xyz", index=":")
x = get_descriptors(dset, k=32, cutoff=5.0)
x = torch.tensor(x, device="cuda")
h = 0.015
batch_size = 10000
H = entropy(x, h=h, batch_size=batch_size)
```

#### Computing overlap between datasets

To compute the overlap between two datasets, you can use the `overlap` command-line interface or the API:

```bash
quests overlap test.xyz ref.xyz -o results.json
```

This command will compute the overlap between the environments in test.xyz and ref.xyz, and save the results to results.json.

Using the API:

```python
from ase.io import read
from quests.descriptor import get_descriptors
from quests.entropy import delta_entropy

test = read("test.xyz", index=":")
ref = read("ref.xyz", index=":")

k, cutoff = 32, 5.0
x1 = get_descriptors(test, k=k, cutoff=cutoff)
x2 = get_descriptors(ref, k=k, cutoff=cutoff)

h = 0.015  # bandwidth
eps = 1e-5  # threshold for overlap
delta = delta_entropy(x1, x2, h=h)
overlap = (delta < eps).mean()

print(f"Overlap value: {overlap:.4f}")
```

This example computes the overlap between two datasets using a bandwidth of 0.015 and an overlap threshold of 1e-3. The overlap is defined as the fraction of environments where the delta entropy is below the threshold.

#### Obtaining a Learning Curve

To generate a learning curve using the command line interface, you can use the `learning_curve` command.
This command computes the entropy at different dataset fractions, allowing you to see how the entropy changes as you include more data:

```bash
quests learning_curve dataset.xyz -o learning_curve_results.json
```

This command will:
1. Use the default fractions (0.1 to 0.9 in steps of 0.1)
2. Compute the entropy for each fraction by randomly sampling environments
3. Run the computation 3 times for each fraction (default value)
4. Save the results in a JSON file named `learning_curve_results.json`

You can customize the command with various options:

- `-f` or `--fractions`: Specify custom fractions (e.g., `-f 0.2,0.4,0.6,0.8`)
- `-n` or `--num_runs`: Set the number of runs for each fraction (e.g., `-n 5`)
- `-b` or `--bandwidth`: Set the bandwidth for entropy calculation (e.g., `-b 0.015`)

A more customized command might look like this:

```bash
quests learning_curve dataset.xyz -f 0.2,0.4,0.6,0.8 -n 5 -c 5.0 -k 32 -b 0.015 -o custom_learning_curve.json
```

This will compute the learning curve for fractions 0.2, 0.4, 0.6, and 0.8, running each fraction 5 times, with a cutoff of 5.0 Å, 32 neighbors, and a bandwidth of 0.015.

The resulting JSON file will contain detailed information about the learning curve, including the entropy values for each fraction and run, as well as the mean and standard deviation of the entropy for each fraction.

#### Compressing datasets with information theory

To compress an atomistic dataset, you can use the `compress` command-line interface or the API:

```bash
quests compress dataset.xyz -m msc -s 0.5 -o results.json
```

This command will compress the dataset using the `msc` method, creating a target dataset with 50% of the size of the original, and saving the metrics and indices of the downselected structures to the `results.json` file.
If you specify an xyz file as an output, it will instead directly generate the dataset.

Using the API with the traditional QUESTS descriptor:

```python
from ase.io import read
from quests.compression import DatasetCompressor

dset = read("dataset.xyz", index=":")

# uses the default QUESTS descriptor and parameters:
# k, cutoff, h = 32, 5.0, 0.015
compressor = DatasetCompressor(dset)

# gets the indices of the structures in the dataset to reduce it to
# 50% of its size using the method `msc`.
# Available methods: `random`, `mean_fps`, `fps`, `k_means`, `msc`
selected = compressor.get_indices(method="msc", size=0.5)

# afterwards, one can use the selected indices to obtain the information
# theoretical metrics about the compressed dataset:
summary = compressor.get_summary(selected)
print(summary)
```

You can use another, custom-made descriptor to compress the dataset as well.
The argument to `DatasetCompression` should be a function that takes an ASE Atoms object with N atoms and returns an `(N, d)` matrix of per-atom descriptors:

```python
from ase.io import read
from quests.compression import DatasetCompressor

dset = read("dataset.xyz", index=":")

descriptor_fn = lambda atoms: your_custom_function(atoms, **kwargs)
compressor = DatasetCompressor(dset, descriptor_fn=descriptor_fn, bandwidth=your_custom_bandwidth)

selected = compressor.get_indices(method="msc", size=0.5)

summary = compressor.get_summary(selected)
print(summary)
```

### Demonstration

One example demonstrating the use of QUESTS for computing the entropy of the Carbon GAP-20 dataset is provided under the folder `examples`.
The script automatically downloads the dataset and shows how to compute the entropy.
Please run this script only after following the installation instructions.

This first example reproduces the first part of Fig. 2c of the manuscript.
Computing the entropies takes a few minutes on a MacBook Pro M3 with 16 threads (default used by numba).

### Manuscript data

Data and notebooks to reproduce the results from the paper are available on Zenodo and GitHub at the following links:


- Data: [![Data](https://zenodo.org/badge/DOI/10.5281/zenodo.15025644.svg)](https://doi.org/10.5281/zenodo.15025644)
- Notebooks: [GitHub](https://github.com/digital-synthesis-lab/2025-quests-data) [![DOI](https://zenodo.org/badge/947665775.svg)](https://doi.org/10.5281/zenodo.15026064)

### Citing

If you use QUESTS in a publication, please cite the following paper:

```bibtex
@article{schwalbekoda2025information,
    title = {Model-free estimation of completeness, uncertainties, and outliers in atomistic machine learning using information theory},
    author = {Schwalbe-Koda, Daniel and Hamel, Sebastien and Sadigh, Babak and Zhou, Fei and Lordi, Vincenzo},
    year = {2025},
    journal = {Nature Communications},
    url = {https://doi.org/10.1038/s41467-025-59232-0},
    doi = {10.1038/s41467-025-59232-0},
    volume={16},
    pages={4014},
}
```
## License

The QUESTS software is distributed under the following license: BSD-3

All new contributions must be made under this license.

SPDX: BSD-3-Clause

## Acknowledgements

This software package was initially produced under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344, with support from LLNL's LDRD program under tracking codes 22-ERD-055 and 23-SI-006.

Code released as LLNL-CODE-858914
