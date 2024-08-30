Installation
============

You can install QUESTS using pip:

.. code-block:: bash

    pip install quests

For the latest development version, you can install directly from the GitHub repository:

.. code-block:: bash

    pip install git+https://github.com/dskoda/quests.git

Requirements
------------

QUESTS requires Python 3.7 or later. The main dependencies are:

- NumPy
- SciPy
- Numba
- ASE (Atomic Simulation Environment)

These dependencies will be automatically installed when you install QUESTS using pip.

Optional Dependencies
---------------------

For GPU support, you'll need:

- PyTorch
- CUDA (for GPU acceleration)

To install QUESTS with GPU support:

.. code-block:: bash

    pip install quests[gpu]

Note that GPU support requires a CUDA-capable GPU and the appropriate CUDA toolkit installed on your system.
