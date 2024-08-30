Welcome to QUESTS's documentation!
==================================

QUESTS (Quick Uncertainty and Entropy via STructural Similarity) provides model-free uncertainty and entropy estimation methods for interatomic potentials.
Among the methods, we propose a structural descriptor based on k-nearest neighbors that:

1. Is fast to compute, as it uses only distances between atoms within an environment.
Because the computation of descriptors is efficiently parallelized, generation of descriptors for 1.5M environments takes about 3 seconds on 56 threads (tested against Intel Xeon CLX-8276L CPUs).
2. Can be used to analyze datasets for atomistic machine learning, providing quantities such as dataset entropy, diversity, information gap, and others.
3. Is shown to recover many useful properties of information theory, and can be used to inform dataset compression

This package also contains tools to interface with other representations and packages.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

API Reference
=============

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
