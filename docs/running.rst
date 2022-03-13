.. _running:

Running the Pipeline
====================

After installation of both ``pycrires`` and ``EsoRex`` (see :ref:`installation` section), we are now ready to start processing CRIRES data with the pipeline!

Downloading data
----------------

First, we need to download a dataset by using the `CRIRES Raw Data Query Form <http://archive.eso.org/wdb/wdb/eso/crires/form>`_ that accesses the `ESO Science Archive Facility <http://archive.eso.org/cms.html>`_. It is recommended to process data that has been obtained with a single observing block (OB), which comprises typically one hour of telescope time.

There are various ways to search for the data with the query form. For example, we can for the science data (i.e. setting *"DPR CATG"* keyword to *"SCIENCE"*) of a specific target name and observing night. After clicking *"Search"*, the required data files are selected and downloaded with *"Request marked datasets"*. On the next screen, we can use *"Run association"* to find the associated calibration files. Finally, the data is downloaded by either generating a shell script or downloading the data directly as ZIP archive.

.. important::
  It is important to download the *"Associated raw calibrations"* (i.e. not the processed calibrations) from the ESO archive. Processing of the calibration data occurs by the pipeline while the use of processed calibrations is currently not supported.

After downloading the raw data, the compressed FITS files (i.e. ending with *.fits.Z*) should be uncompressed, for example with the ``uncompress`` on macOS. Furthermore, the folder that contains the raw data should be renamed to ``raw``, which is a required for initializing the working folder later on.

Processing data
---------------

The ``pycrires`` package provides a Python wrapper for the CRIRES+ and MolecFit recipes of ``EsoRex``. The pipeline functionalities are all embedded in the :class:`~pycrires.pipeline.Pipeline` class.

The instance of ``Pipeline`` requires a path in which the ``raw`` folder is located. This path will be used as working environment in which all the pipeline data will be stored, such as input and configuration files for the recipes and the reduced data products.

After creating the ``Pipeline`` object, we can run all the methods for processing both the calibration and science data. Each method calls a single recipes of the ``EsoRex`` pipeline and produces output which is typically used by the next method.

A configuration file is generated when calling a ``Pipeline`` method for the first time. If needed, the configuration file can be adjusted, which will then be used when rerunning the same method.

.. code-block:: python

  import pycrires

  pipeline = pycrires.Pipeline('./')
  pipeline.rename_files()
  pipeline.extract_header()
  pipeline.cal_dark(verbose=False)
  pipeline.util_calib(calib_type='flat', verbose=False)
  pipeline.util_trace(plot_trace=True, verbose=False)
  pipeline.util_slit_curv(plot_trace=False, verbose=False)
  pipeline.util_extract(calib_type='flat', verbose=False)
  pipeline.util_normflat(verbose=False)
  pipeline.util_calib(calib_type='une', verbose=False)
  pipeline.util_extract(calib_type='une', verbose=False)
  pipeline.util_genlines(verbose=False)
  pipeline.util_wave(calib_type='une', poly_deg=0, wl_err=0.1, verbose=False)
  pipeline.util_wave(calib_type='une', poly_deg=2, wl_err=0.03, verbose=False)
  pipeline.util_calib(calib_type='fpet', verbose=False)
  pipeline.util_extract(calib_type='fpet', verbose=False)
  pipeline.util_wave(calib_type='fpet', poly_deg=4, wl_err=0.01, verbose=False)
  pipeline.obs_nodding(verbose=False)
  pipeline.run_skycalc(pwv=1.)
  pipeline.correct_wavelengths(nod_ab='A', create_plots=True)
  pipeline.plot_spectra(nod_ab='A', telluric=True, corrected=True, file_id=0)
  pipeline.clean_folder(keep_product=False)
