QUESTS Documentation
====================

Welcome to the documentation for QUESTS (Quick Uncertainty and Entropy via STructural Similarity).

QUESTS provides model-free uncertainty and entropy estimation methods for interatomic potentials. It offers a structural descriptor based on k-nearest neighbors that:

1. Is fast to compute, using only distances between atoms within an environment.
2. Can be used to analyze datasets for atomistic machine learning, providing quantities such as dataset entropy, diversity, information gap, and others.
3. Recovers many useful properties of information theory and can be used to inform dataset compression.

This package also contains tools to interface with other representations and packages.

Learn more how to :doc:`install <installation>` the package, or learn more how to use it under the :doc:`quickstart <quickstart>`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   installation
   quickstart
   license

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   :hidden:

   modules

======
Citing
======

If you use quests in your project, please consider citing the `publication <https://arxiv.org/abs/2404.12367>`_ describing the software:

.. code-block:: bibtex

    @article{schwalbekoda2024information,
        title = {Information theory unifies atomistic machine learning, uncertainty quantification, and materials thermodynamics},
        author = {Schwalbe-Koda, Daniel and Hamel, Sebastien and Sadigh, Babak and Zhou, Fei and Lordi, Vincenzo},
        year = {2024},
        journal = {arXiv:2404.12367},
        url = {https://arxiv.org/abs/2404.12367},
    }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
