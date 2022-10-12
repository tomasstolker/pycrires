.. _running:

Running the pipeline
====================

After installation of both ``pycrires`` and ``EsoRex`` (see :ref:`installation` section), we are now ready to start processing `CRIRES+ <https://www.eso.org/sci/facilities/paranal/instruments/crires.html>`_ data with the pipeline!

Downloading data
----------------

First, we need to download a dataset by using the `CRIRES Raw Data Query Form <http://archive.eso.org/wdb/wdb/eso/crires/form>`_ that accesses the `ESO Science Archive Facility <http://archive.eso.org/cms.html>`_. It is recommended to process data that has been obtained with a single observing block (OB), which comprises typically one hour of telescope time.

There are various ways to search for the data with the query form. For example, we can search for the science data (i.e. setting *DPR CATG* to *SCIENCE*) of a specific target and observing night. After searching, the relevant data files can be selected and requested from the archive. Then, on the next screen, the associated raw calibration files can be selected. Finally, the data can be either directly downloaded as ZIP archive or through a shell script.

.. important::
  It is important to download the *Associated raw calibrations* (i.e. not the processed calibrations) from the ESO archive. Processing of the calibration data occurs by the pipeline while the use of processed calibrations is currently not supported.

After downloading the raw data, the compressed FITS files (i.e. ending with *.fits.Z*) should be uncompressed (e.g. with ``uncompress`` in a terminal on macOS). The folder that contains the raw data should be renamed to ``raw``, which is a requirement for initializing the working environment in the next section.

Processing data
---------------

The ``pycrires`` package provides a Python wrapper for the `EsoRex <https://www.eso.org/sci/software/cpl/esorex.html>`_ recipes of CRIRES+ and `MolecFit <https://www.eso.org/sci/software/pipelines/skytools/molecfit>`_. The pipeline functionalities are all embedded in the :class:`~pycrires.pipeline.Pipeline` class.

The instance of :class:`~pycrires.pipeline.Pipeline` requires a path in which the ``raw`` folder is located. This path will be used as working environment in which all the pipeline data will be stored, such as input and configuration files for the recipes and the calibrated data products.

After creating a :class:`~pycrires.pipeline.Pipeline` object, we can run all the methods for processing both the calibration and science data. Each method calls a single recipe of the ``EsoRex`` pipeline and produces output which is then typically used by the next method.

A configuration file is generated when calling a :class:`~pycrires.pipeline.Pipeline` method for the first time. If needed, the configuration file can be adjusted, which will then be used when rerunning the same method.

Below, there is an full example for reducing and calibrating the `CRIRES+ <https://www.eso.org/sci/facilities/paranal/instruments/crires.html>`_ data.

.. tip::
  The pipeline will automatically download any missing calibration files (e.g. dark frames with specific DIT). After downloading and uncompressing, it is important to rerun the pipeline such that the calibration files are included with the data reduction.

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
  pipeline.obs_nodding(verbose=False, correct_bad_pixels=True)
  pipeline.plot_spectra(nod_ab='A', telluric=True, corrected=False, file_id=0)
  pipeline.run_skycalc(pwv=1.)

Next, for spatially-resolved targets (e.g. directly imaged exoplanets), there are dedicated methods for extracting 2D spectra (so maintaining the spatial dimension):

.. code-block:: python

  pipeline.custom_extract_2d(nod_ab='A', spatial_sampling=0.059, max_separation=2.0)
  pipeline.fit_gaussian(nod_ab='A', extraction_input='custom_extract_2d')
  pipeline.correct_wavelengths_2d(nod_ab='A', accuracy=0.01, window_length=201,
                                  minimum_strength=0.005, sum_over_spatial_dim=True,
                                  input_folder='fit_gaussian')

Or, for unresolved targets (e.g. transiting exoplanets), the 1D spectra are already extracted by the ``obs_nodding`` method so we only need to apply the additional wavelength correction:

.. code-block:: python

  pipeline.correct_wavelengths(nod_ab='A', create_plots=True)
  pipeline.plot_spectra(nod_ab='A', telluric=True, corrected=True, file_id=0)
  pipeline.export_spectra(nod_ab='A', corrected=True)  

Finally, for removing any intermediate data products and freeing up some disk space:

.. code-block:: python

  pipeline.clean_folder(keep_product=True)
