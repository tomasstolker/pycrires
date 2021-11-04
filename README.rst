*pycrires*
==========

Pipeline for reduction of high-resolution spectroscopy data from VLT/CRIRES+

.. image:: https://img.shields.io/pypi/v/pycrires
   :target: https://pypi.python.org/pypi/pycrires

.. image:: https://img.shields.io/pypi/pyversions/pycrires
   :target: https://pypi.python.org/pypi/pycrires

.. image:: https://github.com/tomasstolker/pycrires/workflows/CI/badge.svg?branch=main
   :target: https://github.com/tomasstolker/pycrires/actions

.. image:: https://img.shields.io/readthedocs/pycrires
   :target: http://pycrires.readthedocs.io

.. image:: https://codecov.io/gh/tomasstolker/pycrires/branch/main/graph/badge.svg?token=3O7YEYIR8C
   :target: https://codecov.io/gh/tomasstolker/

.. image:: https://img.shields.io/codefactor/grade/github/tomasstolker/pycrires
   :target: https://www.codefactor.io/repository/github/tomasstolker/pycrires

.. image:: https://img.shields.io/github/license/tomasstolker/pycrires
   :target: https://github.com/tomasstolker/pycrires/blob/main/LICENSE

*pycrires* is a Python wrapper for running the CRIRES+ recipes of *EsoRex*. The pipeline organizes the raw data, creates SOF and configuration files for *EsoRex*, runs the calibration and science recipes, applies a telluric correction with *Molecfit*, and creates plots of the images and extracted spectra.

Documentation
-------------

Documentation can be found at `http://pycrires.readthedocs.io <http://pycrires.readthedocs.io>`_.

Contributing
------------

Contributions are welcome so please consider forking the repository and creating a pull request. Bug reports can be provided by creating an `issue <https://github.com/tomasstolker/pycrires/issues>`_ on the Github page.

License
-------

Copyright 2021 Tomas Stolker

*pycrires* is distributed under the MIT License. See the LICENSE file for the terms and conditions.
