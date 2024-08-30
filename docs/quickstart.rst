Quickstart
==========

This quickstart guide will show you how to use the QUESTS tool for analyzing structural datasets.

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
