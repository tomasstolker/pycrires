.. _installation:

Installation
============

*pycrires* is compatible with Python 3.7/3.8/3.9 and is available in the `PyPI repository <https://pypi.org/project/pycrires/>`_ and on `Github <https://github.com/tomasstolker/pycrires>`_.

Installation from PyPI
----------------------

The Python package can be installed with the `pip package manager <https://packaging.python.org/tutorials/installing-packages/>`_:

.. code-block:: console

    $ pip install pycrires

Or, to update to the most recent version:

.. code-block:: console

   $ pip install --upgrade pycrires

.. important::
   The pipeline is currently under development so it is best to install the Github version (see below).

Installation from Github
------------------------

Installation from Github can also be done with pip:

.. code-block:: console

    $ pip install git+git://github.com:tomasstolker/pycrires.git

Alternatively, in case you want to look into the code, it is best to clone the repository:

.. code-block:: console

    $ git clone git@github.com:tomasstolker/pycrires.git

Then, the package is installed by running pip in the cloned repository folder:

.. code-block:: console

    $ pip install -e .

New commits can be pulled from Github once a local copy of the repository exists:

.. code-block:: console

    $ git pull origin main

Do you want to make changes to the code? Please fork the `pycrires` repository on the Github page and clone your own fork instead of the main repository. Contributions and pull requests are very welcome (see :ref:`contributing` section).

EsoRex
------

In addition to ``pycrires``, it is required to install EsoRex and the recipes for CRIRES+. Please follow the instructions on the `ESO website <https://www.eso.org/sci/software/pipelines/>`_. There are recipes available for both the old and upgraded CRIRES instrument so it is important to follow the instructions for *CR2RES* instead of *CRIRES*. On macOS, it is most convenient to use `MacPorts <https://www.eso.org/sci/software/pipelines/installation/macports.html>`_ for installing EsoRex and the recipes.

Testing `pycrires`
------------------

The installation can now be tested, for example by starting Python in interactive mode and printing the version number of the installed package:

.. code-block:: python

    >>> import pycrires
    >>> pycrires.__version__
