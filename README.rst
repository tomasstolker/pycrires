*pycrires*
==========

Data reduction pipeline for VLT/CRIRES+

.. container::

    |PyPI Status| |Python Versions| |CI Status| |Docs Status| |Code Coverage| |Code Quality| |License|

*pycrires* is a Python wrapper for running the CRIRES+ recipes of *EsoRex*.

The pipeline organizes the raw data, creates SOF and configuration files for *EsoRex*, runs the calibration and science recipes, improves the wavelength solution, and creates plots of the images and extracted spectra.

Additionally, there are dedicated routines for the extraction, calibration, and detection of spatially-resolved objects such as directly imaged planets.

For spatially resolved objects, the telluric lines can typically be corrected with the stellar spectrum. Otherwise, it is possible to use an empirical modeling approach with the recipes of *MolecFit*. The pipeline interface of *pycrires* provides functionalities for both cases.

Documentation
-------------

Documentation can be found at `http://pycrires.readthedocs.io <http://pycrires.readthedocs.io>`_.

Attribution
-----------

Please cite `Stolker & Landman (2023) <https://ui.adsabs.harvard.edu/abs/2023ascl.soft07040S/abstract>`_ when *pycrires* is used in a publication and `Landman et al. (2023) <https://ui.adsabs.harvard.edu/abs/2024A%26A...682A..48L/abstract>`_ specifically when using the dedicated routines for spatially-resolved sources.

Contributing
------------

Contributions are welcome so please consider `forking <https://help.github.com/en/articles/fork-a-repo>`_ the repository and creating a `pull request <https://github.com/tomasstolker/pycrires/pulls>`_. Bug reports and feature requests can be provided by creating an `issue <https://github.com/tomasstolker/pycrires/issues>`_ on the Github page.

License
-------

Copyright 2021-2024 Tomas Stolker & Rico Landman

*pycrires* is distributed under the MIT License. See `LICENSE <https://github.com/tomasstolker/pycrires/blob/main/LICENSE>`_ for the terms and conditions.

.. |PyPI Status| image:: https://img.shields.io/pypi/v/pycrires
   :target: https://pypi.python.org/pypi/pycrires

.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/pycrires
   :target: https://pypi.python.org/pypi/pycrires

.. |CI Status| image:: https://github.com/tomasstolker/pycrires/actions/workflows/main.yml/badge.svg
   :target: https://github.com/tomasstolker/pycrires/actions

.. |Docs Status| image:: https://img.shields.io/readthedocs/pycrires
   :target: http://pycrires.readthedocs.io

.. |Code Coverage| image:: https://codecov.io/gh/tomasstolker/pycrires/branch/main/graph/badge.svg?token=LSSCPMJ5JH
   :target: https://codecov.io/gh/tomasstolker/pycrires

.. |Code Quality| image:: https://img.shields.io/codefactor/grade/github/tomasstolker/pycrires
   :target: https://www.codefactor.io/repository/github/tomasstolker/pycrires

.. |License| image:: https://img.shields.io/github/license/tomasstolker/pycrires
   :target: https://github.com/tomasstolker/pycrires/blob/main/LICENSE
