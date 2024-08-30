Quickstart
==========

This quickstart guide will show you how to use the QUESTS tool for analyzing structural datasets.

Using QUESTS with the command line interface
--------------------------------------------


Once installed, you can use the ``quests`` command to perform different analyses. For example, to compute the entropy of any dataset (the input can be anything that ASE reads, including xyz files), you can use the ``quests entropy`` command:

.. code-block:: bash

    quests entropy dump.lammpstrj --bandwidth 0.015

For subsampling the dataset and avoiding using the entire dataset, use the ``entropy_sampler`` example:

.. code-block:: bash

    quests entropy_sampler dataset.xyz --batch_size 20000 -s 100000 -n 3

``-s`` specifies the number of sampled environments, ``-n`` specifies how many runs will be computed (for statistics).


Currently, the following commands are available:

* ``make_descriptors``: computes the QUESTS descriptors and saves them into a numpy ``npz`` file. Useful when performing different analysis for large files.
* ``entropy``: computes the information entropy of a dataset
* ``entropy_sampler``: computes the information entropy of a random fraction of the dataset. The fraction is given as an argument to the command line.
* ``learning_curve``: computes the information entropy of a dataset for a list of fractions, thus creating a learning curve.
* ``dH``: computes the values of differential entropy dH of a test dataset with respect to a reference set.
* ``approx_dH``: computes the **approximate** values of differential entropy dH of a test dataset with respect to a reference set. Requires additional arguments to set up the approximation of neighbors in a vector database.
* ``overlap``: computes the overlap between datasets using the dH analysis.
* ``bandwidth``: estimates the bandwidth as a function of volume according to a physics-defined rule. Useful only if meaningful relationships are to be extracted across datasets.


For additional help with these commands, please use ``quests --help``, or ``quests <command> --help`` for specific help with ``<command>`` (e.g., ``quests entropy --help``).

Creating descriptors
--------------------

To start using the QUESTS package, you can create the representation for a dataset as follows:

.. code-block:: python

    from ase.io import read
    from quests.descriptor import get_descriptors

    # Load your dataset with ASE
    dset = read("dataset.xyz", index=":")

    # create descriptors with k=32 neighbors and a cutoff of 5 Å
    x = get_descriptors(dset, k=32, cutoff=5.0)

In the example above, ``x`` will be a matrix of shape (N, 63), where ``N`` is the number of atoms of the entire dataset.
By default, QUESTS concatenates all the environments into a single matrix.
If you want to generate descriptors for separate systems, you can generate them using a loop:

.. code-block:: python

    from ase.io import read
    from quests.descriptor import get_descriptors

    # Load your dataset with ASE
    dset = read("dataset.xyz", index=":")

    # create descriptors with k=32 neighbors and a cutoff of 5 Å
    xs = [
        x = get_descriptors(atoms, k=32, cutoff=5.0)
        for atoms in dset
    ]

The function ``get_descriptors`` automatically computes the descriptors for periodic and non-periodic systems with different functions.
The length of the descriptors is computed as ``2 * k - 1``, where ``k`` is the number of nearest-neighbors for each atom.

Computing the entropy and diversity
-----------------------------------

Using the descriptors ``x`` as computed above (or any other descriptor), one can compute the entropy using the following function:

.. code-block:: python

    from quests.entropy import perfect_entropy, diversity

    h = 0.015
    batch_size = 10000
    H = perfect_entropy(x, h=h, batch_size=batch_size)
    D = diversity(x, h=h, batch_size=batch_size)

The entropy and diversity are being computed using a Gaussian kernel (default) with bandwidth of 0.015 1/Å and batch size of 10,000.
The metric used by default to compute the distance between two descriptors is the Euclidean one.
The ``batch_size`` is the maximum size of the batch used to compute the distances between descriptors.
For low-memory systems, it helps to use a smaller batch size, though the computation can be slightly slower.

In the example above, the entropy and diversity are given in nats, the units obtained with the natural log, and are simply a float.
To assess what is the maximum possible entropy or diversity that can be achieved in a dataset, it suffices to compute ``np.log(x)``.

Computing differential entropies
--------------------------------

Differential entropies require creating separate descriptors for the datasets:

.. code-block:: python

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

The resulting ``dH`` is an ``np.ndarray`` of size ``len(y)``. Each element ``dH[i]`` is the differential entropy of ``y[i]`` with respect to the dataset ``x``.

If the reference dataset ``x`` is very large and the values of ``dH`` are used for uncertainty quantification (UQ), then one can obtain an upper bound for ``dH`` with its approximation:

.. code-block:: python

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

In the examples above, ``n`` and ``graph_neighbors`` are hyperparameters of the vector database used to create the approximation.
QUESTS uses `pynndescent <https://github.com/lmcinnes/pynndescent>`_ for the vector database, which is a fast implementation of such vector databases with a friendly Python interface.

Computing dataset overlaps
--------------------------

In the manuscript, we often compute a dataset overlap using the values of ``dH``. Using the commands above, the dataset overlap is simply an additional line of code after the calculation of ``dH``:

.. code-block:: python

    dH = delta_entropy(y, x, h=0.015)
    eps = 1e-5
    overlap = (delta < dH).mean()

The small value ``eps`` is used for better numerical stability of the comparison, as the values of ``dH`` can be quite close to 0.

Performing the computations above using GPUs
--------------------------------------------

To accelerate the computation of entropy of datasets, one can use PyTorch to compute the entropy of a system.
This can be done after installing the optional dependencies for this repository (see :doc:`installation instructions <installation>`)
The syntax of the entropy, diversity, and so on, as computed with PyTorch, is identical to the ones above.
Instead of loading the functions from the ``quests.entropy`` module, however, you should load them from ``quests.gpu.entropy``.
The descriptors remain the same - as of now, creating descriptors using GPUs is not supported.
Note that this constraint requires the descriptors to be generated using the traditional routes, and later converted into a ``torch.tensor``.
The example below illustrates this process:

.. code-block:: python

    import torch
    from ase.io import read
    from quests.descriptor import get_descriptors
    from quests.gpu.entropy import perfect_entropy

    dset = read("dataset.xyz", index=":")
    x = get_descriptors(dset, k=32, cutoff=5.0)
    x = torch.tensor(x, device="cuda")
    h = 0.015
    batch_size = 10000
    H = perfect_entropy(x, h=h, batch_size=batch_size)

In the example above, setting a larger batch size will increase the speed of the calculation, but also use more memory.
Set this value judiciously.
