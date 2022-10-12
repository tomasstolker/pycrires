.. _installation:

Installation
============

``pycrires`` is compatible with `Python <https://www.python.org>`_ versions 3.7/3.8/3.9/3.10 and is available from `PyPI <https://pypi.org/project/pycrires/>`_ and `Github <https://github.com/tomasstolker/pycrires>`_.

Installation from PyPI
----------------------

The Python package can be installed from `PyPI <https://pypi.org/project/pycrires/>`_ with the `pip package manager <https://packaging.python.org/tutorials/installing-packages/>`_:

.. code-block:: console

    $ pip install pycrires

Or, to update to the most recent version:

.. code-block:: console

   $ pip install --upgrade pycrires


Installation from Github
------------------------

Using pip
^^^^^^^^^

Installation from `Github <https://github.com/tomasstolker/pycrires>`_ is also possible with ``pip``:

.. code-block:: console

    $ pip install git+https://github.com/tomasstolker/pycrires.git

Cloning the repository
^^^^^^^^^^^^^^^^^^^^^^

Alternatively, in case you want to look into the code, it is best to clone the repository:

.. code-block:: console

    $ git clone git@github.com:tomasstolker/pycrires.git

Then, the package is installed by running ``pip`` in the local repository folder:

.. code-block:: console

    $ pip install -e .

New commits can be pulled from Github once a local copy of the repository exists:

.. code-block:: console

    $ git pull origin main

Do you want to make changes to the code? Please fork the ``pycrires`` repository on the Github page and clone your own fork instead of the main repository. Contributions in the form of pull requests are welcome (see :ref:`about` section).

EsoRex
------

In addition to ``pycrires``, it is required to manually install `EsoRex <https://www.eso.org/sci/software/pipelines>`_, the `CRIRES+ recipes <https://www.eso.org/sci/software/pipelines/cr2res/cr2res-pipe-recipes.html>`_, and the `Molecfit recipes <https://www.eso.org/sci/software/pipelines/molecfit/molecfit-pipe-recipes.html>`_. There are recipes available for both the old and upgraded CRIRES instrument so it is important to follow the instructions for *CR2RES* instead of *CRIRES*. On macOS, it is most convenient to use `MacPorts <https://www.eso.org/sci/software/pipelines/installation/macports.html>`_ for installing both ``EsoRex`` and the recipes.

Testing `pycrires`
------------------

The installation can now be tested, for example by starting Python in interactive mode and printing the version number of the installed package:

.. code-block:: python

    >>> import pycrires
    >>> pycrires.__version__
