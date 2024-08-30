Quickstart
==========

This quickstart guide will show you how to use the QUESTS tool for analyzing structural datasets.

Using QUESTS with the command line interface
--------------------------------------------


Once installed, you can use the `quests` command to perform different analyses. For example, to compute the entropy of any dataset (the input can be anything that ASE reads, including xyz files), you can use the `quests entropy` command:

```bash
quests entropy dump.lammpstrj --bandwidth 0.015
```

For subsampling the dataset and avoiding using the entire dataset, use the `entropy_sampler` example:

```bash
quests entropy_sampler dataset.xyz --batch_size 20000 -s 100000 -n 3
```

`-s` specifies the number of sampled environments, `-n` specifies how many runs will be computed (for statistics).


Currently, the following commands are available:

* `make_descriptors`: computes the QUESTS descriptors and saves them into a numpy `npz` file. Useful when performing different analysis for large files.
* `entropy`: computes the information entropy of a dataset
* `entropy_sampler`: computes the information entropy of a random fraction of the dataset. The fraction is given as an argument to the command line.
* `learning_curve`: computes the information entropy of a dataset for a list of fractions, thus creating a learning curve.
* `dH`: computes the values of differential entropy dH of a test dataset with respect to a reference set.
* `approx_dH`: computes the **approximate** values of differential entropy dH of a test dataset with respect to a reference set. Requires additional arguments to set up the approximation of neighbors in a vector database.
* `overlap`: computes the overlap between datasets using the dH analysis.
* `bandwidth`: estimates the bandwidth as a function of volume according to a physics-defined rule. Useful only if meaningful relationships are to be extracted across datasets.


For additional help with these commands, please use `quests --help`, or `quests <command> --help` for specific help with `<command>` (e.g., `quests entropy --help`).

Analyzing a structural dataset
------------------------------

Here's a simple example of how to use QUESTS to analyze a structural dataset:

.. code-block:: python

    import numpy as np
    from ase.io import read
    from quests import descriptor_pbc
    from quests.entropy import compute_entropy

    # Load your dataset
    atoms_list = read("dataset.xyz", index=":")

    # Compute descriptors
    descriptors = []
    for atoms in atoms_list:
        xyz = atoms.get_positions()
        cell = atoms.get_cell()
        d = descriptor_pbc(xyz, cell)
        descriptors.append(d)

    # Stack descriptors
    X = np.vstack(descriptors)

    # Compute entropy
    S = compute_entropy(X)
    print(f"Dataset entropy: {S:.2f}")

This example demonstrates how to:

1. Load a dataset of atomic structures.
2. Compute descriptors for each structure.
3. Calculate the entropy of the dataset.

For more advanced usage and additional features, please refer to the API documentation.
