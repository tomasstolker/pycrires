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
   The pipeline is under development so currently it is best to install the Github version (see below).

Installation from Github
------------------------

Installation from Github is done directly with pip:

.. code-block:: console

    $ pip install git+git://github.com:tomasstolker/pycrires.git

Or by cloning the repository:

.. code-block:: console

    $ git clone git@github.com:tomasstolker/pycrires.git

And running the setup script to install the package and dependencies:

.. code-block:: console

    $ python setup.py install

.. important::
   If an error occurs when running ``setup.py`` then possibly ``pip`` needs to be updated to the latest version:

   .. code-block:: console

       $ pip install --upgrade pip

Alternatively to running the ``setup.py`` file, the folder where ``pycrires`` is located can also be added to the ``PYTHONPATH`` environment variable such that the package is found by Python. The command may depend on the OS that is used, but is typically something like:

.. code-block:: console

    $ export PYTHONPATH=$PYTHONPATH:/path/to/pycrires

In that case, it is also needed to manually install the dependencies:

.. code-block:: console

    $ pip install -r requirements.txt

New commits can be pulled from Github once a local copy of the repository exists:

.. code-block:: console

    $ git pull origin main

Do you want to make changes to the code? Please fork the `pycrires` repository on the Github page and clone your own fork instead of the main repository. Contributions and pull requests are very welcome (see :ref:`contributing` section).

Testing `pycrires`
------------------

The installation can now be tested, for example by starting Python in interactive mode and printing the version number of the installed package:

.. code-block:: python

    >>> import pycrires
    >>> pycrires.__version__
