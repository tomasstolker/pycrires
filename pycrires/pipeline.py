"""
Module with the pipeline functionalities of ``pycrires``.
"""

import json
import logging
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import urllib.request
import warnings

from typing import Optional

import numpy as np
import pandas as pd
import skycalc_ipy

from astropy import time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.eso import Eso
from matplotlib import pyplot as plt
from scipy import interpolate, ndimage, optimize, signal
from skimage.restoration import inpaint
from typeguard import typechecked

import pycrires


log_book = logging.getLogger(__name__)


class Pipeline:
    """
    Class for the data reduction pipeline. Each method creates a
    configuration file with default values for the `EsoRex` recipe
    that it will run. If needed, these parameters can be adjusted
    before rerunning a method.
    """

    @typechecked
    def __init__(self, path: str) -> None:
        """
        Parameters
        ----------
        path : str
            Path of the main reduction folder. The main folder should
            contain a subfolder called ``raw`` where the raw data
            (both science and calibration) from the ESO archive are
            stored.

        Returns
        -------
        NoneType
            None
        """

        self._print_section(
            "Pipeline for VLT/CRIRES+", bound_char="=", extra_line=False
        )

        # Absolute path of the main reduction folder
        self.path = pathlib.Path(path).resolve()

        print(f"Data reduction folder: {self.path}")

        # Create attributes with the file paths

        self.header_file = pathlib.Path(self.path / "header.csv")
        self.excel_file = pathlib.Path(self.path / "header.xlsx")
        self.json_file = pathlib.Path(self.path / "files.json")

        # Read or create the CSV file with header data

        if self.header_file.is_file():
            print("Reading header data from header.csv")
            self.header_data = pd.read_csv(self.header_file)

        else:
            print("Creating header DataFrame")
            self.header_data = pd.DataFrame()

        # Read or create the JSON file with filenames for SOF

        if self.json_file.is_file():
            print("Reading filenames and labels from files.json")

            with open(self.json_file, "r", encoding="utf-8") as json_file:
                self.file_dict = json.load(json_file)

        else:
            print("Creating dictionary for filenames")
            self.file_dict = {}

        # Create directory for raw files

        self.raw_folder = pathlib.Path(self.path / "raw")

        if not os.path.exists(self.raw_folder):
            os.makedirs(self.raw_folder)

        # Create directory for calibration files

        self.calib_folder = pathlib.Path(self.path / "calib")

        if not os.path.exists(self.calib_folder):
            os.makedirs(self.calib_folder)

        # Create directory for product files

        self.product_folder = pathlib.Path(self.path / "product")

        if not os.path.exists(self.product_folder):
            os.makedirs(self.product_folder)

        # Create directory for configuration files

        self.config_folder = pathlib.Path(self.path / "config")

        if not os.path.exists(self.config_folder):
            os.makedirs(self.config_folder)

        # Test if EsoRex is installed

        if shutil.which("esorex") is None:
            warnings.warn(
                "Esorex is not accessible from the command line. "
                "Please make sure that the ESO pipeline is correctly "
                "installed and included in the PATH variable."
            )

        else:
            # Print the available recipes for CRIRES+ and Molecfit

            esorex = ["esorex", "--recipes"]

            with subprocess.Popen(
                esorex, cwd=self.config_folder, stdout=subprocess.PIPE, encoding="utf-8"
            ) as proc:
                output, _ = proc.communicate()

            print("\nAvailable EsoRex recipes for CRIRES+:")

            for item in output.split("\n"):
                if item.replace(" ", "")[:7] == "cr2res_":
                    print(f"   -{item}")

            print("\nAvailable EsoRex recipes for Molecfit:")

            for item in output.split("\n"):
                if item.replace(" ", "")[:9] == "molecfit_":
                    print(f"   -{item}")

        # Check if there is a new version available

        try:
            contents = urllib.request.urlopen(
                "https://pypi.org/pypi/pycrires/json", timeout=1.0
            ).read()

            data = json.loads(contents)
            latest_version = data["info"]["version"]

        except (urllib.error.URLError, socket.timeout):
            latest_version = None

        if latest_version is not None and pycrires.__version__ != latest_version:
            print(f"\nA new version ({latest_version}) is available!")
            print("Want to stay informed about updates?")
            print("Please have a look at the Github page:")
            print("https://github.com/tomasstolker/pycrires")

    @staticmethod
    @typechecked
    def _print_section(
        sect_title: str,
        bound_char: str = "-",
        extra_line: bool = True,
        recipe_name: Optional[str] = None,
    ) -> None:
        """
        Internal method for printing a section title.

        Parameters
        ----------
        sect_title : str
            Section title.
        bound_char : str
            Boundary character for around the section title.
        extra_line : bool
            Extra new line at the beginning.
        recipe_name : str, None
            Optional name of the `EsoRex` recipe that is used.

        Returns
        -------
        NoneType
            None
        """

        if extra_line:
            print("\n" + len(sect_title) * bound_char)
        else:
            print(len(sect_title) * bound_char)

        print(sect_title)
        print(len(sect_title) * bound_char + "\n")

        if recipe_name is not None:
            print(f"EsoRex recipe: {recipe_name}\n")

    @typechecked
    def _observation_info(self) -> None:
        """
        Internal method for printing some details
        about the observations.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Observation details")

        check_key = {
            "OBS.TARG.NAME": "Target",
            "OBS.PROG.ID": "Program ID",
            "INS.WLEN.ID": "Wavelength setting",
            "INS.WLEN.CWLEN": "Central wavelength (nm)",
            "INS1.DROT.POSANG": "Position angle (deg)",
            "INS.SLIT1.WID": "Slit width (arcsec)",
            "INS.GRAT1.ORDER": "Grating order",
        }

        science_index = self.header_data["DPR.CATG"] == "SCIENCE"

        if "RA" in self.header_data and "DEC" in self.header_data:
            ra_mean = np.mean(self.header_data["RA"][science_index])
            dec_mean = np.mean(self.header_data["DEC"][science_index])

            target_coord = SkyCoord(ra_mean, dec_mean, unit="deg", frame="icrs")

            ra_dec = target_coord.to_string("hmsdms")

            print(f"RA Dec = {ra_dec}")

        for key, value in check_key.items():
            header = self.header_data[key][science_index].to_numpy()

            if isinstance(header[0], str):
                indices = np.where(header is not None)[0]

            else:
                indices = ~np.isnan(header)

                if sum(header == 0.0) != len(header):
                    indices[header == 0.0] = False

            if np.all(header[indices] == header[indices][0]):
                print(f"{value} = {header[0]}")

            else:
                warnings.warn(
                    f"Expecting a single value for {key} but "
                    f"multiple values are found: {header}"
                )

                if isinstance(header[indices][0], np.float64):
                    print(f"{value} = {np.mean(header)}")

        if "OBS.ID" in self.header_data:
            # obs_id = self.header_data['OBS.ID']
            unique_id = pd.unique(self.header_data["OBS.ID"])

            print("\nObservation ID:")

            for item in unique_id:
                if not np.isnan(item):
                    count_files = np.sum(self.header_data["OBS.ID"] == item)

                    if count_files == 1:
                        print(f"   - {item} -> {count_files} file")
                    else:
                        print(f"   - {item} -> {count_files} files")

    @typechecked
    def _export_header(self) -> None:
        """
        Internal method for exporting the ``DataFrame`` with header
        data to a CSV and Excel file.

        Returns
        -------
        NoneType
            None
        """

        # Sort DataFrame by the exposure ID

        self.header_data.sort_values(["DET.EXP.ID"], ascending=True, inplace=True)

        # Write DataFrame to CSV file

        print(f"Exporting DataFrame to {self.header_file.name}")

        self.header_data.to_csv(self.header_file, sep=",", header=True, index=False)

        # Write DataFrame to Excel file

        print(f"Exporting DataFrame to {self.excel_file.name}")

        self.header_data.to_excel(
            self.excel_file, sheet_name="CRIRES", header=True, index=False
        )

        # Read header data from CSV file to set the file indices to the sorted order

        self.header_data = pd.read_csv(self.header_file)

    @typechecked
    def _update_files(self, sof_tag: str, file_name: str) -> None:
        """
        Internal method for updating the dictionary with file
        names and related tag names for the set of files (SOF).

        Parameters
        ----------
        sof_tag : str
            Tag name of ``file_name`` for the set of files (SOF).
        file_name : str
            Absolute path of the file.

        Returns
        -------
        NoneType
            None
        """

        # Print filename and SOF tag

        file_split = file_name.split("/")

        if file_split[-2] == "raw":
            file_print = file_split[-2] + "/" + file_split[-1]
            print(f"   - {file_print} {sof_tag}")

        elif file_split[-3] == "calib":
            file_print = file_split[-3] + "/" + file_split[-2] + "/" + file_split[-1]
            print(f"   - {file_print} {sof_tag}")

        elif file_split[-2] == "product":
            file_print = file_split[-2] + "/" + file_split[-1]
            print(f"   - {file_print} {sof_tag}")

        else:
            print(f"   - {file_name} {sof_tag}")

        # Get FITS header

        if file_name.endswith(".fits"):
            header = fits.getheader(file_name)
        else:
            header = None

        file_dict = {}

        if header is not None and "ESO DET SEQ1 DIT" in header:
            if sof_tag in ["CAL_DARK_MASTER", "CAL_DARK_BPM"]:
                # Use DIT from filename because of issue with
                # cr2res_cal_dark recipe which does not copy
                # the correct header from raw to master dark
                # when processing raw dark with multiple DIT
                file_tmp = file_name.split("/")[-1]
                file_tmp = file_tmp.split("_")[-2]
                file_tmp = file_tmp.split("x")[-2]

                if "." in file_tmp:
                    decimal = len(file_tmp.split(".")[-1])
                    if decimal == 5:
                        for key, value in self.file_dict["DARK"].items():
                            if float(file_tmp) == round(value["DIT"], 5):
                                file_dict["DIT"] = value["DIT"]
                                break
                else:
                    file_dict["DIT"] = float(file_tmp)

            else:
                file_dict["DIT"] = header["ESO DET SEQ1 DIT"]
        else:
            file_dict["DIT"] = None

        if header is not None and "ESO INS WLEN ID" in header:
            file_dict["WLEN"] = header["ESO INS WLEN ID"]
        else:
            file_dict["WLEN"] = None

        if sof_tag in self.file_dict:
            if file_name not in self.file_dict[sof_tag]:
                self.file_dict[sof_tag][file_name] = file_dict
        else:
            self.file_dict[sof_tag] = {file_name: file_dict}

    @typechecked
    def _create_config(
        self, eso_recipe: str, pipeline_method: str, verbose: bool
    ) -> None:
        """
        Internal method for creating a configuration file with default
        values for a specified `EsoRex` recipe. Also check if `EsorRex`
        is found and raise an error otherwise.

        Parameters
        ----------
        eso_recipe : str
            Name of the `EsoRex` recipe.
        pipeline_method : str
            Name of the ``Pipeline`` method.
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        config_file = self.config_folder / f"{pipeline_method}.rc"

        if shutil.which("esorex") is None:
            raise RuntimeError(
                "Esorex is not accessible from the command line. "
                "Please make sure that the ESO pipeline is correctly "
                "installed and included in the PATH variable."
            )

        if not os.path.exists(config_file):
            print()

            esorex = ["esorex", f"--create-config={config_file}", eso_recipe]

            if verbose:
                stdout = None
            else:
                stdout = subprocess.DEVNULL

                print(
                    f"Creating configuration file: config/{pipeline_method}.rc",
                    end="",
                    flush=True,
                )

            subprocess.run(esorex, cwd=self.config_folder, stdout=stdout, check=True)

            # Open config file and adjust some parameters

            with open(config_file, "r", encoding="utf-8") as open_config:
                config_text = open_config.read()

            if eso_recipe == "cr2res_cal_dark":
                # Defaults empirically determined by Thomas Marquart:
                # YJHK bands: bpm_method = GLOBAL, bpm_kappa = 0.5
                # LM bands: bpm_method = LOCAL, bpm_kappa = 6.0
                config_text = config_text.replace(
                    "cr2res.cr2res_cal_dark.bpm_method=DEFAULT",
                    "cr2res.cr2res_cal_dark.bpm_method=GLOBAL",
                )

                config_text = config_text.replace(
                    "cr2res_cal_dark.bpm_kappa=-1.0",
                    "cr2res_cal_dark.bpm_kappa=0.5",
                )

            elif eso_recipe == "cr2res_util_calib":
                config_text = config_text.replace(
                    "cr2res.cr2res_util_calib.collapse=NONE",
                    "cr2res.cr2res_util_calib.collapse=MEAN",
                )

                if pipeline_method == "util_calib_une":
                    # From the manual: If there remain detector
                    # features that correlate with the detector
                    # columns in the combined frames, as has
                    # sometimes been found to be the case,
                    # --subtract_nolight_rows can be set to TRUE.
                    # This makes use of the bottom 40 rows of the
                    # detectors which are intentionally baffled
                    # to not receive light. A vertical median over
                    # these rows is calculated, and the result
                    # subtracted from the image row-by-row.
                    config_text = config_text.replace(
                        "cr2res.cr2res_util_calib.subtract_nolight_rows=FALSE",
                        "cr2res.cr2res_util_calib.subtract_nolight_rows=TRUE",
                    )

            elif eso_recipe == "cr2res_util_extract":
                config_text = config_text.replace(
                    "cr2res.cr2res_util_extract.smooth_slit=2.0",
                    "cr2res.cr2res_util_extract.smooth_slit=3.0",
                )

                if pipeline_method == "util_extract_flat":
                    config_text = config_text.replace(
                        "cr2res.cr2res_util_extract.smooth_spec=0.0",
                        "cr2res.cr2res_util_extract.smooth_spec=2.0e-7",
                    )

                elif pipeline_method == "util_extract_2d":
                    config_text = config_text.replace(
                        "cr2res.cr2res_util_extract.oversample=5",
                        "cr2res.cr2res_util_extract.oversample=8",
                    )

            elif eso_recipe == "cr2res_util_wave":
                config_text = config_text.replace(
                    "cr2res.cr2res_util_wave.fallback_input_wavecal=FALSE",
                    "cr2res.cr2res_util_wave.fallback_input_wavecal=TRUE",
                )

                if pipeline_method == "util_wave_une":
                    config_text = config_text.replace(
                        "cr2res.cr2res_util_wave.wl_method=UNSPECIFIED",
                        "cr2res.cr2res_util_wave.wl_method=XCORR",
                    )

                elif pipeline_method == "util_wave_fpet":
                    config_text = config_text.replace(
                        "cr2res.cr2res_util_wave.wl_method=UNSPECIFIED",
                        "cr2res.cr2res_util_wave.wl_method=ETALON",
                    )

            elif eso_recipe == "cr2res_obs_nodding":
                config_text = config_text.replace(
                    "cr2res.cr2res_obs_nodding.extract_oversample=7",
                    "cr2res.cr2res_obs_nodding.extract_oversample=8",
                )

            elif eso_recipe == "molecfit_model":
                config_text = config_text.replace(
                    "USE_INPUT_KERNEL=TRUE",
                    "USE_INPUT_KERNEL=FALSE",
                )

                # indices = np.where(self.header_data["DPR.CATG"] == "SCIENCE")[0]
                # wlen_id = self.header_data["INS.WLEN.ID"][indices[0]]

                config_text = config_text.replace(
                    "LIST_MOLEC=NULL",
                    "LIST_MOLEC=H2O,CO2,CO,CH4,O2",
                )

                config_text = config_text.replace(
                    "FIT_MOLEC=NULL",
                    "FIT_MOLEC=1,1,1,1,1",
                )

                config_text = config_text.replace(
                    "REL_COL=NULL",
                    "REL_COL=1.0,1.0,1.0,1.0,1.0",
                )

                config_text = config_text.replace(
                    "MAP_REGIONS_TO_CHIP=1",
                    "MAP_REGIONS_TO_CHIP=NULL",
                )

                config_text = config_text.replace(
                    "COLUMN_LAMBDA=lambda",
                    "COLUMN_LAMBDA=WAVE",
                )

                config_text = config_text.replace(
                    "COLUMN_FLUX=flux",
                    "COLUMN_FLUX=SPEC",
                )

                config_text = config_text.replace(
                    "COLUMN_DFLUX=NULL",
                    "COLUMN_DFLUX=ERR",
                )

                config_text = config_text.replace(
                    "PIX_SCALE_VALUE=0.086",
                    "PIX_SCALE_VALUE=0.059",
                )

                config_text = config_text.replace(
                    "FTOL=1e-10",
                    "FTOL=1e-2",
                )

                config_text = config_text.replace(
                    "XTOL=1e-10",
                    "XTOL=1e-2",
                )

                config_text = config_text.replace(
                    "CHIP_EXTENSIONS=FALSE",
                    "CHIP_EXTENSIONS=TRUE",
                )

                config_text = config_text.replace(
                    "FIT_WLC=0",
                    "FIT_WLC=1",
                )

                config_text = config_text.replace(
                    "WLC_N=1",
                    "WLC_N=0",
                )

                config_text = config_text.replace(
                    "WLC_CONST=-0.05",
                    "WLC_CONST=0.0",
                )

                # config_text = config_text.replace(
                #     "FIT_RES_LORENTZ=TRUE",
                #     "FIT_RES_LORENTZ=FALSE",
                # )

                config_text = config_text.replace(
                    "VARKERN=FALSE",
                    "VARKERN=TRUE",
                )

                config_text = config_text.replace(
                    "CONTINUUM_N=0",
                    "CONTINUUM_N=1",
                )

                config_text = config_text.replace(
                    "CONTINUUM_CONST=1.0",
                    "CONTINUUM_CONST=1000.0",
                )

            elif eso_recipe == "molecfit_calctrans":
                config_text = config_text.replace(
                    "USE_INPUT_KERNEL=TRUE",
                    "USE_INPUT_KERNEL=FALSE",
                )

                config_text = config_text.replace(
                    "CHIP_EXTENSIONS=FALSE",
                    "CHIP_EXTENSIONS=TRUE",
                )

            elif eso_recipe == "molecfit_correct":
                config_text = config_text.replace(
                    "CHIP_EXTENSIONS=FALSE",
                    "CHIP_EXTENSIONS=TRUE",
                )

            with open(config_file, "w", encoding="utf-8") as open_config:
                open_config.write(config_text)

            if not verbose:
                print(" [DONE]")

    @typechecked
    def _download_archive(self, dpr_type: str, det_dit: Optional[float]):
        """
        Internal method for downloading data from the ESO Science
        Archive.

        Parameters
        ----------
        dpr_type : str
            The ``DPR.TYPE`` of the data that should be downloaded.
        det_dit : float, None
            The detector integration time (DIT) in case
            ``dpr_type="DARK"``. Can be set to ``None`` otherwise.

        Returns
        -------
        NoneType
            None
        """

        if shutil.which("esorex") is None:
            return

        while True:
            if det_dit is None:
                download = input(
                    f"Could not find data with DPR.TYPE={dpr_type}. "
                    f"Try to download from the ESO Science Archive "
                    f"Facility (y/n)? "
                )

            else:
                download = input(
                    f"There is not {dpr_type} data with DIT = "
                    f"{det_dit} s. Try to download from the "
                    f"ESO Science Archive Facility (y/n)? "
                )

            if download in ["y", "n", "Y", "N"]:
                break

        if download in ["y", "Y"]:
            indices = np.where(self.header_data["DPR.CATG"] == "SCIENCE")[0]

            # Wavelength setting
            wlen_id = self.header_data["INS.WLEN.ID"][indices[0]]

            # Observing date
            date_obs = self.header_data["DATE-OBS"][indices[0]]

            # Sequence of five days before and after the observation
            time_steps = [0.0, -1.0, 1.0, -2.0, 2, -3.0, 3.0, -4.0, 4.0, -5.0, 5.0]
            obs_time = time.Time(date_obs) + np.array(time_steps) * u.day

            # Login at ESO archive

            user_name = input(
                "What is your username to login to the ESO User Portal Services? "
            )

            eso = Eso()
            eso.login(user_name)

            # Query the ESO archive until the data is found

            data_found = False

            for obs_item in obs_time:
                column_filters = {
                    "night": obs_item.value[:10],
                    "dp_type": dpr_type,
                    "ins_wlen_id": wlen_id,
                }

                if det_dit is not None:
                    column_filters["det_dit"] = det_dit

                # eso.query_instrument('crires', help=True)
                table = eso.query_instrument("crires", column_filters=column_filters)

                if table is not None:
                    print(table)

                    data_files = eso.retrieve_data(
                        table["DP.ID"],
                        destination=self.raw_folder,
                        continuation=False,
                        with_calib="none",
                        request_all_objects=False,
                        unzip=False,
                    )

                    print("\nThe following files have been downloaded:")
                    for data_item in data_files:
                        print(f"   - {data_item}")

                    data_found = True
                    break

            if data_found:
                print(
                    "\nThe requested data has been successfully downloaded. "
                    "Please go to the folder with the raw data to uncompress "
                    "any files ending with the .Z extension. Afterwards, "
                    "please rerun the pipeline to ensure that the new "
                    "calibration data is included."
                )
                sys.exit(0)

            else:
                warnings.warn(
                    f"The requested data was not found in the ESO archive. "
                    f"Please download the DPR.TYPE={dpr_type} data manually "
                    "at http://archive.eso.org/wdb/wdb/eso/crires/form. "
                    f"Continuing the data reduction without the {dpr_type} "
                    f"data."
                )

        else:
            warnings.warn(
                f"For best results, please download the suggested "
                f"DPR.TYPE={dpr_type} data from the ESO archive at "
                f"http://archive.eso.org/wdb/wdb/eso/crires/form. "
                f"Continuing without the {dpr_type} data now."
            )

    @typechecked
    def _plot_image(self, file_type: str, fits_folder: str):
        """
        Internal method for plotting the data of a specified file type.

        Parameters
        ----------
        file_type : str
            The file type of which the data should be plotted.
        fits_folder : str
            Folder in which to search for the FITS files of type
            ``file_type``. The argument should be specified relative
            to the main reduction folder (e.g. "calib/cal_dark" or
            "calib/util_calib_flat").

        Returns
        -------
        NoneType
            None
        """

        if file_type in self.file_dict:
            print(f"\nPlotting {file_type}:")

            for item in self.file_dict[file_type]:
                file_name = item.split("/")

                if f"{file_name[-3]}/{file_name[-2]}" != fits_folder:
                    continue

                print(f"   - calib/{file_name[-2]}/{file_name[-1][:-4]}png")

                with fits.open(item) as hdu_list:
                    plt.figure(figsize=(10, 3.5))

                    header = hdu_list[0].header

                    dit = header["HIERARCH ESO DET SEQ1 DIT"]
                    ndit = header["HIERARCH ESO DET NDIT"]

                    for i in range(3):
                        plt.subplot(1, 3, i + 1)

                        data = hdu_list[f"CHIP{i+1}.INT1"].data
                        data = np.nan_to_num(data)

                        vmin, vmax = np.percentile(data, (1, 99))

                        plt.imshow(
                            data, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax
                        )
                        plt.title(f"Detector {i+1}", fontsize=9)
                        plt.minorticks_on()

                    plt.suptitle(
                        f"{file_name[-1]}, {file_type}, range = [{vmin:.1f}:{vmax:.1f}], "
                        f"DIT = {dit}, NDIT = {ndit}",
                        y=0.95,
                        fontsize=10,
                    )

                    plt.tight_layout()
                    plt.savefig(f"{item[:-4]}png", dpi=300)
                    plt.clf()
                    plt.close()

        else:
            warnings.warn(f"Could not find {file_type} files to plot.")

    @typechecked
    def _plot_trace(self, dpr_catg: str):
        """
        Internal method for plotting the raw data of a specified
        ``DPR.CATG`` together with the traces of the spectral orders.
        The content of this method has been adapted from the
        ``cr2res_show_trace_curv.py`` code that is included with the
        EsoRex pipeline for CRIRES+ (see
        https://www.eso.org/sci/software/pipelines/).

        Parameters
        ----------
        dpr_catg : str
            The ``DPR.CATG`` of which the raw data should be plotted.

        Returns
        -------
        NoneType
            None
        """

        print(f"\nPlotting {dpr_catg} with trace:")

        if "UTIL_SLIT_CURV_TW" in self.file_dict:
            trace_file = list(self.file_dict["UTIL_SLIT_CURV_TW"].keys())[0]
        else:
            trace_file = list(self.file_dict["UTIL_TRACE_TW"].keys())[0]

        with fits.open(trace_file) as hdu_list:
            trace_data = [hdu_list[1].data, hdu_list[2].data, hdu_list[3].data]

        indices = self.header_data["DPR.CATG"] == dpr_catg

        x_range = np.arange(2048)

        for file_item in self.header_data[indices]["ORIGFILE"]:
            print(f"   - raw/{file_item[:-5]}_trace.png")

            with fits.open(f"{self.path}/raw/{file_item}") as hdu_list:
                plt.figure(figsize=(10, 3.5))

                header = hdu_list[0].header

                dit = header["HIERARCH ESO DET SEQ1 DIT"]
                ndit = header["HIERARCH ESO DET NDIT"]

                for k in range(3):
                    plt.subplot(1, 3, k + 1)

                    data = hdu_list[f"CHIP{k+1}.INT1"].data

                    vmin, vmax = np.percentile(data, (5, 95))

                    plt.imshow(
                        data, origin="lower", cmap="plasma", vmin=vmin, vmax=vmax
                    )

                    plt.xlim(0, 2048)
                    plt.ylim(0, 2048)

                    plt.minorticks_on()

                    for t in trace_data[k]:
                        upper = np.polyval(t["Upper"][::-1], x_range)
                        plt.plot(x_range, upper, ls=":", lw=0.8, color="white")

                        lower = np.polyval(t["Lower"][::-1], x_range)
                        plt.plot(x_range, lower, ls=":", lw=0.8, color="white")

                        middle = np.polyval(t["All"][::-1], x_range)
                        plt.plot(x_range, middle, ls="--", lw=0.8, color="white")

                        select_trace = trace_data[k][
                            trace_data[k]["order"] == t["Order"]
                        ]

                        i1 = select_trace["Slitfraction"][:, 1]
                        i2 = select_trace["All"]

                        coeff = [
                            np.interp(0.5, i1, i2[:, k]) for k in range(i2.shape[1])
                        ]

                        for i in range(30, 2048, 200):
                            ew = [int(middle[i] - lower[i]), int(upper[i] - middle[i])]
                            x = np.zeros(ew[0] + ew[1] + 1)
                            y = np.arange(-ew[0], ew[1] + 1).astype(float)

                            # Evaluate the curvature polynomials to get coefficients
                            a = np.polyval(t["SlitPolyA"][::-1], i)
                            b = np.polyval(t["SlitPolyB"][::-1], i)
                            c = np.polyval(t["SlitPolyC"][::-1], i)
                            yc = np.polyval(coeff[::-1], i)

                            # Shift polynomials to the local frame of reference
                            a = a - i + yc * b + yc * yc * c
                            b += 2 * yc * c

                            for j, y_item in enumerate(y):
                                x[j] = i + y_item * b + y_item**2 * c

                            plt.plot(x, y + middle[i], ls="-", lw=0.8, color="white")

                            plt.text(
                                500,
                                middle[1024],
                                f"order: {t['order']}\ntrace: {t['TraceNb']}",
                                color="white",
                                ha="left",
                                va="center",
                                fontsize=7.0,
                            )

                plt.suptitle(
                    f"{file_item}, {dpr_catg}, range = [{vmin:.1f}:{vmax:.1f}], "
                    f"DIT = {dit}, NDIT = {ndit}",
                    y=0.95,
                    fontsize=10,
                )

                plt.tight_layout()
                plt.savefig(f"{self.path}/raw/{file_item[:-5]}_trace.png", dpi=300)
                plt.clf()
                plt.close()

    @typechecked
    def rename_files(self) -> None:
        """
        Method for renaming the files from ``ARCFILE`` to ``ORIGFILE``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Renaming files")

        raw_files = sorted(pathlib.Path(self.path / "raw").glob("*.fits"))

        n_total = 0
        n_renamed = 0

        acq_files = []
        science_files = []
        calib_files = []

        for item in raw_files:
            header = fits.getheader(item)

            if "ESO DPR CATG" in header:
                dpr_catg = header["ESO DPR CATG"]

                if dpr_catg == "SCIENCE":
                    science_files.append(item)

                elif dpr_catg in "CALIB":
                    calib_files.append(item)

                elif dpr_catg == "ACQUISITION":
                    acq_files.append(item)

                else:
                    warnings.warn(
                        f"The DPR.CATG with value {dpr_catg} "
                        f"has not been recognized."
                    )

            if "ARCFILE" in header and "ORIGFILE" in header:
                if item.name == header["ARCFILE"]:
                    os.rename(item, item.parent / header["ORIGFILE"])
                    n_renamed += 1

            elif "ARCFILE" not in header:
                warnings.warn(
                    f"The ARCFILE keyword is not found in the header of {item.name}"
                )

            elif "ORIGFILE" not in header:
                warnings.warn(
                    f"The ORIGFILE keyword is not found in "
                    f"the header of {item.name}"
                )

            n_total += 1

        print("Science data:\n")
        for item in science_files:
            print(f"   - {item.name}")

        print("\nCalibration data:\n")
        for item in calib_files:
            print(f"   - {item.name}")

        print("\nAcquisition data:\n")
        for item in acq_files:
            print(f"   - {item.name}")

        print(f"\nTotal tumber of FITS files: {n_total}")
        print(f"Number of renamed files: {n_renamed}")

    @typechecked
    def extract_header(self) -> None:
        """
        Method for extracting relevant header data from the FITS files
        and storing these in a ``DataFrame``. The data will also be
        exported to a CSV and Excel file.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Extracting FITS headers")

        # Create a new DataFrame
        self.header_data = pd.DataFrame()
        print("Creating new DataFrame...\n")

        key_file = os.path.dirname(__file__) + "/keywords.txt"
        keywords = np.genfromtxt(key_file, dtype="str", delimiter=",")

        raw_files = pathlib.Path(self.path / "raw").glob("*.fits")

        header_dict = {}
        for key_item in keywords:
            header_dict[key_item] = []

        for file_item in raw_files:
            header = fits.getheader(file_item)

            for key_item in keywords:
                if key_item in header:
                    header_dict[key_item].append(header[key_item])
                else:
                    header_dict[key_item].append(None)

        for key_item in keywords:
            column_name = key_item.replace(" ", ".")
            column_name = column_name.replace("ESO.", "")

            self.header_data[column_name] = header_dict[key_item]

        self._export_header()

        indices = np.where(self.header_data["DPR.CATG"] == "SCIENCE")[0]

        if len(indices) > 0:
            self._observation_info()
        else:
            warnings.warn(
                "Could not find any DPR.CATG=SCIENCE data "
                "so there will not be any details printed "
                "about the observations."
            )

    @typechecked
    def select_bpm(self, wlen_id, dit_select) -> Optional[str]:
        """
        Method for selecting the bad pixel map (BPM)
        of the requested DIT. An adjusted is made
        for long exposure in the :math:`K` band, of
        which some orders can be affected by the
        thermal continuum.

        Parameters
        ----------
        wlen_id : str
            Wavelength setting of the science exposures.
        dit_select : float
            Detector integration time (DIT) for
            which the BPM should be selected.

        Returns
        -------
        str, None
            Filename of the BPM. A ``None`` is returned
            if a BPM was not found.
        """

        bpm_dit = set()
        for key, value in self.file_dict["CAL_DARK_BPM"].items():
            bpm_dit.add(value["DIT"])

        file_found = False
        bpm_file = None

        warn_msg = (
            "The thermal continuum becomes visible in "
            "the reddest orders of long DARK exposures "
            "obtained in the K band. This affects the "
            "identification of bad pixels with the GLOBAL "
            "method. A BPM that was derived from a shorter "
            "exposure will therefore be selected instead."
        )

        for key, value in self.file_dict["CAL_DARK_BPM"].items():
            if not file_found:
                if wlen_id[0] == "K" and dit_select > 10.0:
                    if value["DIT"] < 10.0:
                        warnings.warn(warn_msg)
                        bpm_file = key
                        file_found = True

                elif dit_select == value["DIT"]:
                    bpm_file = key
                    file_found = True

        return bpm_file

    @typechecked
    def run_skycalc(self, pwv: float = 3.5) -> None:
        """
        Method for running the Python wrapper of SkyCalc
        (see https://skycalc-ipy.readthedocs.io).

        Parameters
        ----------
        pwv : float
            Precipitable water vapor (default: 3.5) that is used for
            the telluric spectrum. This value will impact the depth
            of the telluric lines which can be seen when plotting
            the spectra with
            :func:`~pycrires.pipeline.Pipeline.plot_spectra` while
            setting ``telluric=True``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Run SkyCalc")

        # Create output folder

        output_dir = self.calib_folder / "run_skycalc"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Indices with SCIENCE frames

        science_index = np.where(self.header_data["DPR.CATG"] == "SCIENCE")[0]

        # Requested PWV for observations

        pwv_req = self.header_data["OBS.WATERVAPOUR"][science_index[0]]
        print(f"Requested PWV with observations (mm) = {pwv_req}\n")

        # Setup SkyCalc object

        print("SkyCalc settings:")

        sky_calc = skycalc_ipy.SkyCalc()

        mjd_start = self.header_data["MJD-OBS"][science_index[0]]
        ra_mean = np.mean(self.header_data["RA"][science_index])
        dec_mean = np.mean(self.header_data["DEC"][science_index])

        sky_calc.get_almanac_data(
            ra=ra_mean,
            dec=dec_mean,
            date=None,
            mjd=mjd_start,
            observatory="paranal",
            update_values=True,
        )

        print(f"  - MJD = {mjd_start:.2f}")
        print(f"  - RA (deg) = {ra_mean:.2f}")
        print(f"  - Dec (deg) = {dec_mean:.2f}")

        # See https://skycalc-ipy.readthedocs.io/en/latest/GettingStarted.html
        sky_calc["msolflux"] = 130

        indices = self.header_data["INS.WLEN.ID"].notna()
        wlen_id = self.header_data["INS.WLEN.ID"][indices].iloc[0]

        if wlen_id[0] == "Y":
            sky_calc["wmin"] = 500.0  # (nm)
            sky_calc["wmax"] = 1500.0  # (nm)

        elif wlen_id[0] == "J":
            sky_calc["wmin"] = 800.0  # (nm)
            sky_calc["wmax"] = 2000.0  # (nm)

        elif wlen_id[0] == "H":
            sky_calc["wmin"] = 1000.0  # (nm)
            sky_calc["wmax"] = 2500.0  # (nm)

        elif wlen_id[0] == "K":
            sky_calc["wmin"] = 1500.0  # (nm)
            sky_calc["wmax"] = 3000.0  # (nm)

        elif wlen_id[0] == "L":
            sky_calc["wmin"] = 2500.0  # (nm)
            sky_calc["wmax"] = 4500.0  # (nm)

        else:
            raise NotImplementedError(
                f"The wavelength range for {wlen_id} is not yet implemented."
            )

        sky_calc["wgrid_mode"] = "fixed_spectral_resolution"
        sky_calc["wres"] = 2e5
        sky_calc["pwv"] = pwv

        print(f"  - Wavelength range (nm) = {sky_calc['wmin']} - {sky_calc['wmax']}")
        print(f"  - lambda / Dlambda = {sky_calc['wres']}")
        print(f"  - Airmass = {sky_calc['airmass']:.2f}")
        print(f"  - PWV (mm) = {sky_calc['pwv']}\n")

        # Get telluric spectra from SkyCalc

        print("Get telluric spectrum with SkyCalc...", end="", flush=True)

        temp_file = self.calib_folder / "run_skycalc/skycalc_temp.fits"

        sky_spec = sky_calc.get_sky_spectrum(filename=temp_file)

        print(" [DONE]\n")

        # Convolve spectra

        slit_width = self.header_data["INS.SLIT1.NAME"][science_index[0]]

        if slit_width == "w_0.2":
            spec_res = 100000.0
        elif slit_width == "w_0.4":
            spec_res = 50000.0
        else:
            raise ValueError(f"Slit width {slit_width} not recognized.")

        print(f"Slit width = {slit_width}")
        print(f"Smoothing spectrum to R = {spec_res}\n")

        sigma_lsf = 1.0 / spec_res / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        spacing = np.mean(
            2.0
            * np.diff(sky_spec["lam"])
            / (sky_spec["lam"][1:] + sky_spec["lam"][:-1])
        )

        sigma_lsf_gauss_filter = sigma_lsf / spacing

        sky_spec["flux"] = ndimage.gaussian_filter(
            sky_spec["flux"], sigma=sigma_lsf_gauss_filter, mode="nearest"
        )

        sky_spec["trans"] = ndimage.gaussian_filter(
            sky_spec["trans"], sigma=sigma_lsf_gauss_filter, mode="nearest"
        )

        # Telluric emission spectrum

        print("Storing emission spectrum: calib/run_skycalc/sky_spec.dat")

        emis_spec = np.column_stack((1e3 * sky_spec["lam"], 1e-3 * sky_spec["flux"]))
        header = "Wavelength (nm) - Flux (ph arcsec-2 m-2 s-1 nm-1)"

        out_file = self.calib_folder / "run_skycalc/sky_spec.dat"
        np.savetxt(out_file, emis_spec, header=header)

        # Telluric transmission spectrum

        print("Storing transmission spectrum: calib/run_skycalc/transm_spec.dat")

        transm_spec = np.column_stack((1e3 * sky_spec["lam"], sky_spec["trans"]))
        header = "Wavelength (nm) - Transmission"

        out_file = self.calib_folder / "run_skycalc/transm_spec.dat"
        np.savetxt(out_file, transm_spec, header=header)

    @typechecked
    def cal_dark(self, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_cal_dark``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create master DARK", recipe_name="cr2res_cal_dark")

        indices = self.header_data["DPR.TYPE"] == "DARK"

        # Create output folder

        output_dir = self.calib_folder / "cal_dark"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Check unique DIT

        unique_dit = set()
        for item in self.header_data[indices]["DET.SEQ1.DIT"]:
            unique_dit.add(item)

        if len(unique_dit) == 0:
            print("Unique DIT values: none")
        else:
            print(f"Unique DIT values: {unique_dit}\n")

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(output_dir / "files.sof")

        with open(sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                sof_open.write(f"{self.path}/raw/{item} DARK\n")
                self._update_files("DARK", f"{self.path}/raw/{item}")

        # Check if any dark frames were found

        if "DARK" not in self.file_dict:
            raise RuntimeError(
                "The 'raw' folder does not contain any DPR.TYPE=DARK files."
            )

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_cal_dark", "cal_dark", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "cal_dark.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "cr2res_cal_dark",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with master dark

        print("Output files:")

        fits_files = pathlib.Path(self.path / "calib/cal_dark").glob(
            "cr2res_cal_dark_*master.fits"
        )

        for item in fits_files:
            self._update_files("CAL_DARK_MASTER", str(item))

        # Update file dictionary with bad pixel map

        fits_files = pathlib.Path(self.path / "calib/cal_dark").glob(
            "cr2res_cal_dark_*bpm.fits"
        )

        for item in fits_files:
            self._update_files("CAL_DARK_BPM", str(item))

        # Create plots

        self._plot_image("CAL_DARK_MASTER", "calib/cal_dark")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def cal_flat(self, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_cal_flat``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create master FLAT", recipe_name="cr2res_cal_flat")

        indices = self.header_data["DPR.TYPE"] == "FLAT"

        # Create output folder

        output_dir = self.calib_folder / "cal_flat"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Check unique wavelength settings

        unique_wlen = set()
        for item in self.header_data[indices]["INS.WLEN.ID"]:
            unique_wlen.add(item)

        if len(unique_wlen) > 1:
            raise RuntimeError(
                f"The FLAT files consist of more than "
                f"one wavelength setting: {unique_wlen}. "
                f"Please only include FLAT files with "
                f"the same INS.WLEN.ID as the science "
                f"exposures."
            )

        # Check unique DIT

        unique_dit = set()
        for item in self.header_data[indices]["DET.SEQ1.DIT"]:
            unique_dit.add(item)

        if len(unique_dit) == 0:
            print("Unique DIT values: none")
        else:
            print(f"Unique DIT values: {unique_dit}\n")

        # Wavelength setting

        science_idx = np.where(self.header_data["DPR.CATG"] == "SCIENCE")[0]
        wlen_id = self.header_data["INS.WLEN.ID"][science_idx[0]]

        # Iterate over different DIT values for FLAT

        for dit_item in unique_dit:
            print(f"Creating SOF file for DIT={dit_item}:")

            sof_file = pathlib.Path(self.path / "calib/cal_flat/files.sof")

            with open(sof_file, "w", encoding="utf-8") as sof_open:
                for item in self.header_data[indices]["ORIGFILE"]:
                    index = self.header_data.index[
                        self.header_data["ORIGFILE"] == item
                    ][0]
                    flat_dit = self.header_data.iloc[index]["DET.SEQ1.DIT"]

                    if flat_dit == dit_item:
                        file_path = f"{self.path}/raw/{item}"
                        sof_open.write(f"{file_path} FLAT\n")
                        self._update_files("FLAT", file_path)

                # Find master dark

                file_found = False

                if "CAL_DARK_MASTER" in self.file_dict:
                    for key, value in self.file_dict["CAL_DARK_MASTER"].items():
                        if not file_found and value["DIT"] == dit_item:
                            file_name = key.split("/")[-2:]
                            print(
                                f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_DARK_MASTER"
                            )
                            sof_open.write(f"{key} CAL_DARK_MASTER\n")
                            file_found = True

                if not file_found:
                    self._download_archive("DARK", dit_item)

                # Find bad pixel map

                file_found = False

                if "CAL_DARK_BPM" in self.file_dict:
                    bpm_file = self.select_bpm(wlen_id, dit_item)

                    if bpm_file is not None:
                        file_name = bpm_file.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_DARK_BPM"
                        )
                        sof_open.write(f"{bpm_file} CAL_DARK_BPM\n")
                        file_found = True

                if not file_found:
                    warnings.warn(
                        f"There is not a bad pixel map " f"with DIT = {dit_item} s."
                    )

            # Create EsoRex configuration file if not found

            self._create_config("cr2res_cal_flat", "cal_flat", verbose)

            # Run EsoRex

            print()

            config_file = self.config_folder / "cal_flat.rc"

            esorex = [
                "esorex",
                f"--recipe-config={config_file}",
                f"--output-dir={output_dir}",
                "cr2res_cal_flat",
                sof_file,
            ]

            if verbose:
                stdout = None
            else:
                stdout = subprocess.DEVNULL
                print("Running EsoRex...", end="", flush=True)

            subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

            if not verbose:
                print(" [DONE]\n")

            # Update file dictionary with master flat

            print("Output files:")

            fits_files = pathlib.Path(self.path / "calib").glob(
                "cr2res_cal_flat_*master_flat.fits"
            )

            for item in fits_files:
                self._update_files("CAL_FLAT_MASTER", str(item))

            # Update file dictionary with TraceWave table

            fits_files = pathlib.Path(self.path / "calib").glob(
                "cr2res_cal_flat_*tw.fits"
            )

            for item in fits_files:
                self._update_files("CAL_FLAT_TW", str(item))

            # Create plots

            self._plot_image("CAL_FLAT_MASTER", "calib/cal_flat")

            # Write updated dictionary to JSON file

            with open(self.json_file, "w", encoding="utf-8") as json_file:
                json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def cal_detlin(self, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_cal_detlin``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section(
            "Determine detector linearity", recipe_name="cr2res_cal_detlin"
        )

        indices = self.header_data["DPR.TYPE"] == "FLAT,LAMP,DETCHECK"

        # Create output folder

        output_dir = self.calib_folder / "cal_detlin"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Check unique DIT

        unique_dit = set()
        for item in self.header_data[indices]["DET.SEQ1.DIT"]:
            unique_dit.add(item)

        if len(unique_dit) == 0:
            print("Unique DIT values: none")
        else:
            print(f"Unique DIT values: {unique_dit}\n")

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(output_dir / "files.sof")

        with open(sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                sof_open.write(f"{self.path}/raw/{item} DETLIN_LAMP\n")
                self._update_files("DETLIN_LAMP", f"{self.path}/raw/{item}")

        # Check if any dark frames were found

        if "DETLIN_LAMP" not in self.file_dict:
            raise RuntimeError(
                "The 'raw' folder does not contain any "
                "DPR.TYPE=FLAT,LAMP,DETCHECK files."
            )

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_cal_detlin", "cal_detlin", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "cal_detlin.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "cr2res_cal_detlin",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with CAL_DETLIN_COEFFS file

        print("Output files:")

        fits_file = f"{output_dir}/cr2res_cal_detlin_coeffs.fits"

        self._update_files("CAL_DETLIN_COEFFS", fits_file)

        # Update file dictionary with CAL_DETLIN_BPM file

        fits_file = f"{output_dir}/cr2res_cal_detlin_bpm.fits"

        self._update_files("CAL_DETLIN_BPM", fits_file)

        # Create plots

        self._plot_image("CAL_DETLIN_BPM", "calib/cal_detlin")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def cal_wave(self, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_cal_wave``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Wavelength calibration", recipe_name="cr2res_cal_wave")

        # Create output folder

        output_dir = self.calib_folder / "cal_wave"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(self.path / "calib/cal_wave/files.sof")

        with open(sof_file, "w", encoding="utf-8") as sof_open:
            # Uranium-Neon lamp (UNE) frames

            indices = self.header_data["DPR.TYPE"] == "WAVE,UNE"

            une_found = False

            if sum(indices) == 0:
                warnings.warn(
                    "The 'raw' folder does not contain a DPR.TYPE=WAVE,UNE file."
                )

            elif sum(indices) > 1:
                raise RuntimeError(
                    "More than one WAVE,UNE file "
                    "Please only provided a single "
                    "WAVE,UNE file."
                )

            else:
                une_found = True

            for item in self.header_data[indices]["ORIGFILE"]:
                file_path = f"{self.path}/raw/{item}"
                sof_open.write(f"{file_path} WAVE_UNE\n")
                self._update_files("WAVE_UNE", file_path)

            # Fabry Pérot Etalon (FPET) frames

            indices = self.header_data["DPR.TYPE"] == "WAVE,FPET"

            fpet_found = False

            if sum(indices) == 0:
                indices = self.header_data["OBJECT"] == "WAVE,FPET"

            if sum(indices) == 0:
                warnings.warn(
                    "The 'raw' folder does not contain a DPR.TYPE=WAVE,FPET file."
                )

            elif sum(indices) > 1:
                raise RuntimeError(
                    "More than one WAVE,FPET file "
                    "Please only provided a single "
                    "WAVE,FPET file."
                )

            else:
                fpet_found = True

            for item in self.header_data[indices]["ORIGFILE"]:
                file_path = f"{self.path}/raw/{item}"
                sof_open.write(f"{file_path} WAVE_FPET\n")
                self._update_files("WAVE_FPET", file_path)

            # Find TraceWave file

            file_found = False

            if "UTIL_TRACE_TW" in self.file_dict:
                for key in self.file_dict["UTIL_TRACE_TW"]:
                    if not file_found:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_TRACE_TW"
                        )
                        sof_open.write(f"{key} UTIL_TRACE_TW\n")
                        file_found = True

            if "CAL_WAVE_TW" in self.file_dict:
                for key in self.file_dict["CAL_WAVE_TW"]:
                    if not file_found:
                        file_name = key.split("/")[-2:]
                        print(f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_WAVE_TW")
                        sof_open.write(f"{key} CAL_WAVE_TW\n")
                        file_found = True

            if "CAL_FLAT_TW" in self.file_dict:
                for key in self.file_dict["CAL_FLAT_TW"]:
                    if not file_found:
                        file_name = key.split("/")[-2:]
                        print(f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_FLAT_TW")
                        sof_open.write(f"{key} CAL_FLAT_TW\n")
                        file_found = True

            if not file_found:
                raise RuntimeError("Could not find a TraceWave file.")

            # Find emission lines file

            file_found = False

            if "EMISSION_LINES" in self.file_dict:
                for key in self.file_dict["EMISSION_LINES"]:
                    if not file_found:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} EMISSION_LINES"
                        )
                        sof_open.write(f"{key} EMISSION_LINES\n")
                        file_found = True

            if not file_found:
                raise RuntimeError("Could not find an emission lines file.")

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_cal_wave", "cal_wave", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "cal_wave.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "cr2res_cal_wave",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with UNE wave tables

        print("Output files:")

        if une_found:
            spec_file = f"{self.path}/calib/cal_wave/cr2res_cal_wave_tw_une.fits"
            self._update_files("CAL_WAVE_TW", spec_file)

            spec_file = f"{self.path}/calib/cal_wave/cr2res_cal_wave_wave_map_une.fits"
            self._update_files("CAL_WAVE_MAP", spec_file)

        # Update file dictionary with FPET wave tables

        if fpet_found:
            spec_file = f"{self.path}/calib/cal_wave/cr2res_cal_wave_tw_fpet.fits"
            self._update_files("CAL_WAVE_TW", spec_file)

            spec_file = f"{self.path}/calib/cal_wave/cr2res_cal_wave_wave_map_fpet.fits"
            self._update_files("CAL_WAVE_MAP", spec_file)

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def util_calib(self, calib_type: str, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_util_calib``.

        Parameters
        ----------
        calib_type : str
            Calibration type ("flat", "une", "fpet").
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        if calib_type == "flat":
            self._print_section("Create master FLAT", recipe_name="cr2res_util_calib")

        elif calib_type == "une":
            self._print_section("Create master UNE", recipe_name="cr2res_util_calib")

        elif calib_type == "fpet":
            self._print_section("Create master FPET", recipe_name="cr2res_util_calib")

        else:
            raise RuntimeError("The argument of 'calib_type' is not recognized.")

        # Create output folder

        output_dir = self.calib_folder / f"util_calib_{calib_type}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if calib_type == "flat":
            indices = self.header_data["DPR.TYPE"] == "FLAT"

            # Check unique wavelength settings

            unique_wlen = set()
            for item in self.header_data[indices]["INS.WLEN.ID"]:
                unique_wlen.add(item)

            if len(unique_wlen) > 1:
                raise RuntimeError(
                    f"The FLAT files consist of more than "
                    f"one wavelength setting: {unique_wlen}. "
                    f"Please only include FLAT files with "
                    f"the same INS.WLEN.ID as the science "
                    f"exposures."
                )

        elif calib_type == "une":
            indices = self.header_data["DPR.TYPE"] == "WAVE,UNE"

            if indices.sum() == 0:
                self._download_archive("WAVE,UNE", None)

        elif calib_type == "fpet":
            indices = self.header_data["DPR.TYPE"] == "WAVE,FPET"

            if indices.sum() == 0:
                self._download_archive("WAVE,FPET", None)

        if indices.sum() == 0:
            raise RuntimeError(
                f"Could not find a raw calibration "
                f"file for calib_type={calib_type}."
            )

        # Check unique DIT

        unique_dit = set()
        for item in self.header_data[indices]["DET.SEQ1.DIT"]:
            unique_dit.add(item)

        if len(unique_dit) == 0:
            print("Unique DIT values: none")
        else:
            print(f"Unique DIT values: {unique_dit}\n")

        unique_dit = list(unique_dit)

        # Only process FLAT from a single DIT

        if len(unique_dit) == 1:
            dit_item = unique_dit[0]

        elif len(unique_dit) > 1:
            dit_item = input(
                "Which DIT of the FLAT files should be used "
                "for creating the UTIL_CALIB file?"
            )

            dit_item = float(dit_item)

        science_idx = np.where(self.header_data["DPR.CATG"] == "SCIENCE")[0]
        wlen_id = self.header_data["INS.WLEN.ID"][science_idx[0]]

        print(f"Creating SOF file for DIT={dit_item}:")

        sof_file = pathlib.Path(self.path / f"calib/util_calib_{calib_type}/files.sof")

        with open(sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                index = self.header_data.index[self.header_data["ORIGFILE"] == item][0]
                calib_dit = self.header_data.iloc[index]["DET.SEQ1.DIT"]

                if calib_dit == dit_item:
                    file_path = f"{self.path}/raw/{item}"

                    if calib_type == "flat":
                        sof_open.write(f"{file_path} FLAT\n")
                        self._update_files("FLAT", file_path)

                    elif calib_type == "une":
                        sof_open.write(f"{file_path} WAVE_UNE\n")
                        self._update_files("WAVE_UNE", file_path)

                    elif calib_type == "fpet":
                        sof_open.write(f"{file_path} WAVE_FPET\n")
                        self._update_files("WAVE_FPET", file_path)

            # Find master dark

            file_found = False

            if "CAL_DARK_MASTER" in self.file_dict:
                for key, value in self.file_dict["CAL_DARK_MASTER"].items():
                    if not file_found and value["DIT"] == dit_item:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_DARK_MASTER"
                        )
                        sof_open.write(f"{key} CAL_DARK_MASTER\n")
                        file_found = True

            if not file_found:
                self._download_archive("DARK", dit_item)

            # Find bad pixel map

            file_found = False

            if "CAL_DARK_BPM" in self.file_dict:
                bpm_file = self.select_bpm(wlen_id, dit_item)

                if bpm_file is not None:
                    file_name = bpm_file.split("/")[-2:]
                    print(f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_DARK_BPM")
                    sof_open.write(f"{bpm_file} CAL_DARK_BPM\n")
                    file_found = True

            if not file_found:
                warnings.warn(f"There is not a bad pixel map with DIT = {dit_item} s.")

            # Find UTIL_MASTER_FLAT or CAL_FLAT_MASTER file
            # when DPR.TYPE is WAVE,UNE or WAVE,FPET

            if calib_type in ["une", "fpet"]:
                file_found = False

                if "UTIL_MASTER_FLAT" in self.file_dict:
                    for key, value in self.file_dict["UTIL_MASTER_FLAT"].items():
                        if not file_found:
                            file_name = key.split("/")[-2:]
                            print(
                                f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_MASTER_FLAT"
                            )
                            sof_open.write(f"{key} UTIL_MASTER_FLAT\n")
                            file_found = True

                if "CAL_FLAT_MASTER" in self.file_dict:
                    for key, value in self.file_dict["CAL_FLAT_MASTER"].items():
                        if not file_found:
                            file_name = key.split("/")[-2:]
                            print(
                                f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_FLAT_MASTER"
                            )
                            sof_open.write(f"{key} CAL_FLAT_MASTER\n")
                            file_found = True

                if not file_found:
                    warnings.warn(
                        "The CAL_FLAT_MASTER file is not found in "
                        "the 'calib' folder so continuing without "
                        "applying a master flat."
                    )

            # Find CAL_DETLIN_COEFFS file

            file_found = False

            if "CAL_DETLIN_COEFFS" in self.file_dict:
                for key, value in self.file_dict["CAL_DETLIN_COEFFS"].items():
                    if not file_found:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_DETLIN_COEFFS"
                        )
                        sof_open.write(f"{key} CAL_DETLIN_COEFFS\n")
                        file_found = True

            if not file_found:
                warnings.warn("Could not find CAL_DETLIN_COEFFS.")

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_util_calib", f"util_calib_{calib_type}", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / f"util_calib_{calib_type}.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "cr2res_util_calib",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with UTIL_CALIB file

        print("Output files:")

        fits_file = (
            f"{self.path}/calib/util_calib_{calib_type}"
            + "/cr2res_util_calib_calibrated_collapsed.fits"
        )

        self._update_files("UTIL_CALIB", fits_file)

        # Create plots

        self._plot_image("UTIL_CALIB", f"calib/util_calib_{calib_type}")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def util_trace(self, plot_trace: bool = False, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_util_trace``.

        Parameters
        ----------
        plot_trace : bool
            Plot the traces of the spectral orders on the raw data.
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section(
            "Trace the spectral orders", recipe_name="cr2res_util_trace"
        )

        # Create output folder

        output_dir = self.calib_folder / "util_trace"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(self.path / "calib/util_trace/files.sof")

        # Find UTIL_CALIB file

        file_found = False

        if "UTIL_CALIB" in self.file_dict:
            for key in self.file_dict["UTIL_CALIB"]:
                if not file_found:
                    with open(sof_file, "w", encoding="utf-8") as sof_open:
                        file_name = key.split("/")[-2:]
                        print(f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_CALIB")
                        sof_open.write(f"{key} UTIL_CALIB\n")
                        file_found = True

        # Otherwise use raw FLAT files

        if not file_found:
            indices = self.header_data["DPR.TYPE"] == "FLAT"

            with open(sof_file, "w", encoding="utf-8") as sof_open:
                for item in self.header_data[indices]["ORIGFILE"]:
                    print(f"   - raw/{item} FLAT")
                    sof_open.write(f"{self.path}/raw/{item} FLAT\n")

            warnings.warn(
                "There is not a UTIL_CALIB file so using raw FLAT "
                "files instead. To use the UTIL_CALIB file, please "
                "first run the util_calib method."
            )

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_util_trace", "util_trace", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "util_trace.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "cr2res_util_trace",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with UTIL_TRACE_TW file

        print("Output files:")

        if file_found:
            fits_file = (
                f"{self.path}/calib/util_trace/"
                + "cr2res_util_calib_calibrated_collapsed_tw.fits"
            )

            self._update_files("UTIL_TRACE_TW", fits_file)

        else:
            for item in self.header_data[indices]["ORIGFILE"]:
                fits_file = f"{self.path}/calib/util_trace/{item[:-5]}_tw.fits"
                self._update_files("UTIL_TRACE_TW", fits_file)

        # Create plots

        if plot_trace:
            self._plot_trace("CALIB")
            self._plot_trace("SCIENCE")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def util_slit_curv(self, plot_trace: bool = False, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_util_slit_curv``.

        Parameters
        ----------
        plot_trace : bool
            Plot the traces of the spectral orders on the raw data.
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section(
            "Compute slit curvature", recipe_name="cr2res_util_slit_curv"
        )

        # Create output folder

        output_dir = self.calib_folder / "util_slit_curv"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(self.path / "calib/util_slit_curv/files.sof")

        # Find Fabry Pérot Etalon (FPET) files

        indices = self.header_data["DPR.TYPE"] == "WAVE,FPET"

        if sum(indices) == 0:
            indices = self.header_data["OBJECT"] == "WAVE,FPET"

        if sum(indices) == 0:
            raise RuntimeError(
                "The 'raw' folder does not contain a DPR.TYPE=WAVE,FPET file."
            )

        if sum(indices) > 1:
            raise RuntimeError(
                "More than one WAVE,FPET file is present in the "
                "'raw' folder. Please only provided a single "
                "WAVE,FPET file."
            )

        with open(sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                file_path = f"{self.path}/raw/{item}"
                sof_open.write(f"{file_path} WAVE_FPET\n")
                self._update_files("WAVE_FPET", file_path)

        # Find UTIL_TRACE_TW file

        file_found = False

        if "UTIL_TRACE_TW" in self.file_dict:
            for key in self.file_dict["UTIL_TRACE_TW"]:
                if not file_found:
                    with open(sof_file, "a", encoding="utf-8") as sof_open:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_TRACE_TW"
                        )
                        sof_open.write(f"{key} UTIL_TRACE_TW\n")
                        file_found = True

        else:
            raise RuntimeError(
                "The UTIL_TRACE_TW file is not found in "
                "the 'calib' folder. Please first run "
                "the util_trace method."
            )

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_util_slit_curv", "util_slit_curv", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "util_slit_curv.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "cr2res_util_slit_curv",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with UTIL_SLIT_CURV_TW file

        print("Output files:")

        fits_file = f"{output_dir}/cr2res_util_calib_calibrated_collapsed_tw_tw.fits"

        self._update_files("UTIL_SLIT_CURV_TW", fits_file)

        # Update file dictionary with UTIL_SLIT_CURV_MAP file

        fits_file = f"{output_dir}/cr2res_util_calib_calibrated_collapsed_tw_map.fits"

        self._update_files("UTIL_SLIT_CURV_MAP", fits_file)

        # Create plots

        if plot_trace:
            self._plot_trace("CALIB")
            self._plot_trace("SCIENCE")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def util_extract(self, calib_type: str, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_util_extract``.

        Parameters
        ----------
        calib_type : str
            Calibration type ("flat", "une", or "fpet").
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        if calib_type == "flat":
            self._print_section(
                "Extract FLAT spectrum", recipe_name="cr2res_util_extract"
            )

        elif calib_type == "une":
            self._print_section(
                "Extract UNE spectrum", recipe_name="cr2res_util_extract"
            )

        elif calib_type == "fpet":
            self._print_section(
                "Extract FPET spectrum", recipe_name="cr2res_util_extract"
            )

        else:
            raise RuntimeError("The argument of 'calib_type' is not recognized.")

        # Create output folder

        output_dir = self.calib_folder / f"util_extract_{calib_type}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(
            self.path / f"calib/util_extract_{calib_type}/files.sof"
        )

        # Find UTIL_CALIB file

        file_found = False

        if "UTIL_CALIB" in self.file_dict:
            for key in self.file_dict["UTIL_CALIB"]:
                if not file_found:
                    if key.split("/")[-2] == f"util_calib_{calib_type}":
                        with open(sof_file, "w", encoding="utf-8") as sof_open:
                            file_name = key.split("/")[-2:]
                            print(
                                f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_CALIB"
                            )
                            sof_open.write(f"{key} UTIL_CALIB\n")
                            file_found = True

            if not file_found:
                raise RuntimeError(
                    f"The UTIL_CALIB file for calib_type={calib_type} "
                    f"is not found in the 'calib/util_calib_{calib_type}' "
                    f"folder. Please first run the util_calib method."
                )

        else:
            raise RuntimeError(
                "The UTIL_CALIB file is not found in the 'calib' "
                "folder. Please first run the util_calib method."
            )

        # Find UTIL_TRACE_TW file

        # file_found = False
        #
        # if "UTIL_TRACE_TW" in self.file_dict:
        #     for key, value in self.file_dict["UTIL_TRACE_TW"].items():
        #         if not file_found:
        #             with open(sof_file, "a", encoding="utf-8") as sof_open:
        #                 file_name = key.split("/")[-2:]
        #                 print(f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_TRACE_TW")
        #                 sof_open.write(f"{key} UTIL_TRACE_TW\n")
        #                 file_found = True
        #
        # else:
        #     raise RuntimeError(
        #         "The UTIL_TRACE_TW file is not found in "
        #         "the 'calib' folder. Please first run "
        #         "the util_trace method."
        #     )

        # Find UTIL_SLIT_CURV_TW file

        file_found = False

        if "UTIL_WAVE_TW" in self.file_dict:
            for key in self.file_dict["UTIL_WAVE_TW"]:
                if not file_found and key.split("/")[-2] == "util_wave_une":
                    with open(sof_file, "a", encoding="utf-8") as sof_open:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_SLIT_CURV_TW"
                        )
                        sof_open.write(f"{key} UTIL_SLIT_CURV_TW\n")
                        file_found = True

        if "UTIL_SLIT_CURV_TW" in self.file_dict:
            for key in self.file_dict["UTIL_SLIT_CURV_TW"]:
                if not file_found:
                    with open(sof_file, "a", encoding="utf-8") as sof_open:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_SLIT_CURV_TW"
                        )
                        sof_open.write(f"{key} UTIL_SLIT_CURV_TW\n")
                        file_found = True

        if not file_found:
            raise RuntimeError(
                "The UTIL_SLIT_CURV_TW file is not found "
                "in the 'calib' folder. Please first run "
                "the util_slit_curv method."
            )

        # Create EsoRex configuration file if not found

        self._create_config(
            "cr2res_util_extract", f"util_extract_{calib_type}", verbose
        )

        # Run EsoRex

        print()

        config_file = self.config_folder / f"util_extract_{calib_type}.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "cr2res_util_extract",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with UTIL_EXTRACT_1D file

        print("Output files:")

        fits_file = (
            f"{self.path}/calib/util_extract_{calib_type}/"
            + "cr2res_util_calib_calibrated_collapsed_extr1D.fits"
        )

        self._update_files("UTIL_EXTRACT_1D", fits_file)

        # Update file dictionary with UTIL_SLIT_FUNC file

        fits_file = (
            f"{self.path}/calib/util_extract_{calib_type}/"
            + "cr2res_util_calib_calibrated_collapsed_extrSlitFu.fits"
        )

        self._update_files("UTIL_SLIT_FUNC", fits_file)

        # Update file dictionary with UTIL_SLIT_MODEL file

        fits_file = (
            f"{self.path}/calib/util_extract_{calib_type}/"
            + "cr2res_util_calib_calibrated_collapsed_extrModel.fits"
        )

        self._update_files("UTIL_SLIT_MODEL", fits_file)

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def util_normflat(self, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_util_normflat``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section(
            "Create normalized flat field", recipe_name="cr2res_util_normflat"
        )

        # Create output folder

        output_dir = self.calib_folder / "util_normflat"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(self.path / "calib/util_normflat/files.sof")

        # Find UTIL_CALIB file

        file_found = False

        if "UTIL_CALIB" in self.file_dict:
            for key in self.file_dict["UTIL_CALIB"]:
                if not file_found:
                    with open(sof_file, "w", encoding="utf-8") as sof_open:
                        file_name = key.split("/")[-2:]
                        print(f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_CALIB")
                        sof_open.write(f"{key} UTIL_CALIB\n")
                        file_found = True

        else:
            raise RuntimeError(
                "The UTIL_CALIB file is not found in "
                "the 'calib' folder. Please first run "
                "the util_calib method."
            )

        # Find UTIL_SLIT_MODEL file

        file_found = False

        if "UTIL_SLIT_MODEL" in self.file_dict:
            for key in self.file_dict["UTIL_SLIT_MODEL"]:
                if not file_found:
                    with open(sof_file, "a", encoding="utf-8") as sof_open:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_SLIT_MODEL"
                        )
                        sof_open.write(f"{key} UTIL_SLIT_MODEL\n")
                        file_found = True

        else:
            raise RuntimeError(
                "The UTIL_SLIT_MODEL file is not found in "
                "the 'calib' folder. Please first run "
                "the util_extract method."
            )

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_util_normflat", "util_normflat", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "util_normflat.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "cr2res_util_normflat",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with UTIL_MASTER_FLAT file

        print("Output files:")

        fits_file = f"{self.path}/calib/util_normflat/cr2res_util_normflat_Open_master_flat.fits"
        self._update_files("UTIL_MASTER_FLAT", fits_file)

        # Update file dictionary with UTIL_NORM_BPM file

        fits_file = (
            f"{self.path}/calib/util_normflat/cr2res_util_normflat_Open_master_bpm.fits"
        )
        self._update_files("UTIL_NORM_BPM", fits_file)

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def util_genlines(self, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_util_genlines``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section(
            "Generate calibration lines", recipe_name="cr2res_util_genlines"
        )

        # Create output folder

        output_dir = self.calib_folder / "util_genlines"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Table with emission lines
        # Y/J -> lines_u_sarmiento.txt
        # H/K -> lines_u_redman.txt
        # L/M -> lines_thar.txt

        code_dir = os.path.dirname(os.path.realpath(__file__))

        indices = np.where(self.header_data["DPR.CATG"] == "SCIENCE")[0]

        wavel_set = self.header_data["INS.WLEN.ID"][indices[0]]

        if wavel_set[0] in ["Y", "J"]:
            line_file = code_dir + "/calib_data/lines_u_sarmiento.txt"

        elif wavel_set[0] in ["H", "K"]:
            line_file = code_dir + "/calib_data/lines_u_redman.txt"

        elif wavel_set[0] in ["L", "M"]:
            line_file = code_dir + "/calib_data/lines_thar.txt"

        else:
            raise RuntimeError(
                "Could not find calibration file with "
                "emission lines for the requested "
                "wavelength setting ({wavel_set})."
            )

        # Table with wavelength ranges

        range_file = code_dir + f"/calib_data/{wavel_set}.dat"

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(self.path / "calib/util_genlines/files.sof")

        with open(sof_file, "w", encoding="utf-8") as sof_open:
            sof_open.write(f"{line_file} EMISSION_LINES_TXT\n")
            self._update_files("EMISSION_LINES_TXT", line_file)

            sof_open.write(f"{range_file} LINES_SELECTION_TXT\n")
            self._update_files("LINES_SELECTION_TXT", range_file)

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_util_genlines", "util_genlines", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "util_genlines.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "cr2res_util_genlines",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with EMISSION_LINES files

        print("Output files:")

        fits_file = (
            str(output_dir) + "/" + line_file.split("/")[-1].replace(".txt", ".fits")
        )
        self._update_files("EMISSION_LINES", fits_file)

        indices = np.where(self.header_data["DPR.CATG"] == "SCIENCE")[0]
        wlen_id = self.header_data["INS.WLEN.ID"][indices[0]]

        fits_file = (
            str(output_dir)
            + "/"
            + line_file.split("/")[-1].replace(".txt", f"_{wlen_id}.fits")
        )
        self._update_files("EMISSION_LINES", fits_file)

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def util_wave(
        self,
        calib_type: str,
        poly_deg: int = 0,
        wl_err: float = 0.1,
        verbose: bool = True,
    ) -> None:
        """
        Method for running ``cr2res_util_wave``.

        Parameters
        ----------
        calib_type : str
            Calibration type ("une" or "fpet").
        poly_deg : int
            Polynomial degree for fitting the wavelength solution.
        wl_err : float
            Estimate wavelength error (in nm).
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        if calib_type == "une":
            self._print_section(
                "Determine wavelength solution", recipe_name="cr2res_util_wave"
            )

        elif calib_type == "fpet":
            self._print_section(
                "Refine wavelength solution", recipe_name="cr2res_util_wave"
            )

        else:
            raise RuntimeError("The argument of 'calib_type' is not recognized.")

        # Create output folder

        output_dir = self.calib_folder / f"util_wave_{calib_type}"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(self.path / f"calib/util_wave_{calib_type}/files.sof")

        # Find EMISSION_LINES file

        file_found = False

        if calib_type == "une":

            if "EMISSION_LINES" in self.file_dict:
                for key in self.file_dict["EMISSION_LINES"]:
                    if not file_found:
                        with open(sof_file, "w", encoding="utf-8") as sof_open:
                            file_name = key.split("/")[-2:]
                            print(
                                f"   - calib/{file_name[-2]}/{file_name[-1]} EMISSION_LINES"
                            )
                            sof_open.write(f"{key} EMISSION_LINES\n")
                            file_found = True

            else:
                raise RuntimeError(
                    "The EMISSION_LINES file is not found in "
                    "the 'calib/genlines' folder. Please first "
                    "run the util_genlines method."
                )

        else:
            with open(sof_file, "w", encoding="utf-8") as sof_open:
                pass

        # Find UTIL_CALIB file

        file_found = False

        if "UTIL_EXTRACT_1D" in self.file_dict:
            for key in self.file_dict["UTIL_EXTRACT_1D"]:
                if (
                    not file_found
                    and key.split("/")[-2] == f"util_extract_{calib_type}"
                ):
                    with open(sof_file, "a", encoding="utf-8") as sof_open:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_EXTRACT_1D"
                        )
                        sof_open.write(f"{key} UTIL_EXTRACT_1D\n")
                        file_found = True

            if not file_found:
                raise RuntimeError(
                    f"The UTIL_EXTRACT_1D file is not found in the "
                    f"'calib/util_extract_{calib_type}' folder. Please first "
                    f"run the util_extract method with calib_type='une'."
                )

        else:
            raise RuntimeError(
                f"The UTIL_EXTRACT_1D file is not found in the "
                f"'calib/util_extract_{calib_type}' folder. Please first "
                f"run the util_extract method with calib_type='une'."
            )

        # Find UTIL_SLIT_CURV_TW file

        if calib_type == "une":
            file_found = False

            if "UTIL_WAVE_TW" in self.file_dict and poly_deg > 0:
                for key in self.file_dict["UTIL_WAVE_TW"]:
                    if not file_found and key.split("/")[-2] == "util_wave_une":
                        with open(sof_file, "a", encoding="utf-8") as sof_open:
                            file_name = key.split("/")[-2:]
                            print(
                                f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_SLIT_CURV_TW"
                            )
                            sof_open.write(f"{key} UTIL_SLIT_CURV_TW\n")
                            file_found = True

            if "UTIL_SLIT_CURV_TW" in self.file_dict:
                for key in self.file_dict["UTIL_SLIT_CURV_TW"]:
                    if not file_found:
                        with open(sof_file, "a", encoding="utf-8") as sof_open:
                            file_name = key.split("/")[-2:]
                            print(
                                f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_SLIT_CURV_TW"
                            )
                            sof_open.write(f"{key} UTIL_SLIT_CURV_TW\n")
                            file_found = True

            if not file_found:
                raise RuntimeError(
                    "The UTIL_SLIT_CURV_TW file is not found "
                    "in the 'calib/util_slit_curv' folder. "
                    "Please first run the util_slit_curv method."
                )

        # Find UTIL_WAVE_TW file

        if calib_type == "fpet":
            file_found = False

            if "UTIL_WAVE_TW" in self.file_dict:
                for key in self.file_dict["UTIL_WAVE_TW"]:
                    if not file_found and key.split("/")[-2] == "util_wave_une":
                        with open(sof_file, "a", encoding="utf-8") as sof_open:
                            file_name = key.split("/")[-2:]
                            print(
                                f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_WAVE_TW"
                            )
                            sof_open.write(f"{key} UTIL_WAVE_TW\n")
                            file_found = True

            else:
                raise RuntimeError(
                    "The UTIL_WAVE_TW file is not found in the "
                    "'calib/util_wave_une' folder. Please first "
                    "run the util_wave method with calib_type='une'."
                )

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_util_wave", f"util_wave_{calib_type}", verbose)

        # Run EsoRex

        if calib_type == "fpet":
            wl_err = -1.0

        if poly_deg == 0:
            keep_high_deg = "TRUE"
        else:
            keep_high_deg = "FALSE"

        print("\nConfiguration:")
        print(f"   - Polynomial degree = {poly_deg}")
        print(f"   - Wavelength error (nm) = {wl_err}")
        print(f"   - Keep higher degrees = {keep_high_deg}\n")

        config_file = self.config_folder / f"util_wave_{calib_type}.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "cr2res_util_wave",
            f"--wl_degree={poly_deg}",
            f"--wl_err={wl_err}",
            f"--keep={keep_high_deg}",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]")

        print()

        # Update file dictionary with UTIL_WAVE_TW file

        print("Output files:")

        fits_file = (
            f"{self.path}/calib/util_wave_{calib_type}/"
            + "cr2res_util_calib_calibrated_collapsed_extr1D_tw.fits"
        )

        self._update_files("UTIL_WAVE_TW", fits_file)

        # Update file dictionary with UTIL_WAVE_MAP file

        fits_file = (
            f"{self.path}/calib/util_wave_{calib_type}/"
            + "cr2res_util_calib_calibrated_collapsed_extr1D_wave_map.fits"
        )

        self._update_files("UTIL_WAVE_MAP", fits_file)

        # Update file dictionary with UTIL_WAVE_LINES_DIAGNOSTICS file

        fits_file = (
            f"{self.path}/calib/util_wave_{calib_type}/"
            + "cr2res_util_calib_calibrated_collapsed_extr1D_lines_diagnostics.fits"
        )

        self._update_files("UTIL_WAVE_LINES_DIAGNOSTICS", fits_file)

        # Update file dictionary with UTIL_WAVE_EXTRACT_1D file

        fits_file = (
            f"{self.path}/calib/util_wave_{calib_type}/"
            + "cr2res_util_calib_calibrated_collapsed_extr1D_extracted.fits"
        )

        self._update_files("UTIL_WAVE_EXTRACT_1D", fits_file)

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def obs_nodding(
        self,
        verbose: bool = True,
        correct_bad_pixels: bool = True,
    ) -> None:
        """
        Method for running ``cr2res_obs_nodding``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.
        correct_bad_pixels : bool
            Correct bad pixels with the bad pixel map and
            ``skimage.restoration.inpaint``. If set to
            ``False``, the bad pixels will remain as NaN
            in the output images.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Process nodding frames", recipe_name="cr2res_obs_nodding")

        indices = self.header_data["DPR.CATG"] == "SCIENCE"

        # Create output folder

        output_dir = self.product_folder / "obs_nodding"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # for i, item in enumerate(self.header_data["DPR.TYPE"]):
        #     if item != "OBJECT":
        #         indices[i] = False

        # Check unique DIT

        # unique_dit = set()
        # for item in self.header_data[indices]["DET.SEQ1.DIT"]:
        #     unique_dit.add(item)
        #
        # print(f"Unique DIT values: {unique_dit}\n")

        # Wavelength setting and DIT

        science_idx = np.where(self.header_data["DPR.CATG"] == "SCIENCE")[0]
        science_wlen = self.header_data["INS.WLEN.ID"][science_idx[0]]
        science_dit = self.header_data["DET.SEQ1.DIT"][science_idx[0]]

        # Count nod positions

        nod_a_exp = (self.header_data["SEQ.NODPOS"] == "A") & (
            self.header_data["DPR.CATG"] == "SCIENCE"
        )

        nod_b_exp = (self.header_data["SEQ.NODPOS"] == "B") & (
            self.header_data["DPR.CATG"] == "SCIENCE"
        )

        nod_a_count = sum(nod_a_exp)
        nod_b_count = sum(nod_b_exp)

        print(f"Number of exposures at nod A: {nod_a_count}")
        print(f"Number of exposures at nod B: {nod_b_count}")

        if nod_a_count != nod_b_count:
            warnings.warn(
                "There is an unequal number of exposures "
                "at nod A and nod B. The pipeline has not "
                "been tested for this so an error or "
                "unexpected results may occur."
            )

        # Create SOF file

        count_exp = 0

        # Iterate over nod A exposures
        for i_row in self.header_data.index[nod_a_exp]:
            print(
                f"\nCreating SOF file for nod pair #{count_exp+1}/{indices.sum()//2}:"
            )
            sof_file = pathlib.Path(output_dir / f"files_{count_exp:03d}.sof")

            sof_open = open(sof_file, "w", encoding="utf-8")

            file_0 = self.header_data["ORIGFILE"][i_row]

            if self.header_data["SEQ.NODPOS"][i_row + 1] == "B":
                # AB pair, so using the next exposure for B
                file_1 = self.header_data["ORIGFILE"][i_row + 1]

            elif self.header_data["SEQ.NODPOS"][i_row - 1] == "B":
                # BA pair, so using the previous exposure for B
                file_1 = self.header_data["ORIGFILE"][i_row - 1]

            else:
                warnings.warn(
                    f"Can not find nod B data to use "
                    f"in combination with the nod A "
                    f"data of {file_0} so will skip "
                    f"this file."
                )

                continue

            file_path_0 = f"{self.path}/raw/{file_0}"
            file_path_1 = f"{self.path}/raw/{file_1}"

            header_0 = fits.getheader(file_path_0)
            # header_1 = fits.getheader(file_path_1)

            if "ESO DPR TECH" in header_0:
                if header_0["ESO DPR TECH"] == "SPECTRUM,NODDING,OTHER":
                    sof_open.write(f"{file_path_0} OBS_NODDING_OTHER\n")
                    self._update_files("OBS_NODDING_OTHER", file_path_0)

                    sof_open.write(f"{file_path_1} OBS_NODDING_OTHER\n")
                    self._update_files("OBS_NODDING_OTHER", file_path_1)

                elif header_0["ESO DPR TECH"] == "SPECTRUM,NODDING,JITTER":
                    sof_open.write(f"{file_path_0} OBS_NODDING_JITTER\n")
                    self._update_files("OBS_NODDING_JITTER", file_path_0)

                    sof_open.write(f"{file_path_1} OBS_NODDING_JITTER\n")
                    self._update_files("OBS_NODDING_JITTER", file_path_1)

            else:
                raise RuntimeError(
                    f"Could not find ESO.DPR.TECH in " f"the header of {file_path_0}."
                )

            # Find UTIL_MASTER_FLAT or CAL_FLAT_MASTER file

            file_found = False

            if "UTIL_MASTER_FLAT" in self.file_dict:
                for key in self.file_dict["UTIL_MASTER_FLAT"]:
                    if not file_found:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_MASTER_FLAT"
                        )
                        sof_open.write(f"{key} UTIL_MASTER_FLAT\n")
                        file_found = True

            if "CAL_FLAT_MASTER" in self.file_dict:
                for key in self.file_dict["CAL_FLAT_MASTER"]:
                    if not file_found:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_FLAT_MASTER"
                        )
                        sof_open.write(f"{key} CAL_FLAT_MASTER\n")
                        file_found = True

            if not file_found:
                warnings.warn("Could not find a master flat.")

            # Find CAL_DARK_BPM file

            file_found = False

            if "CAL_DARK_BPM" in self.file_dict:
                bpm_file = self.select_bpm(science_wlen, science_dit)

                if bpm_file is not None:
                    file_name = bpm_file.split("/")[-2:]
                    print(f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_DARK_BPM")
                    sof_open.write(f"{bpm_file} CAL_DARK_BPM\n")
                    file_found = True

            if not file_found:
                warnings.warn("Could not find a bap pixel map.")

            # Find UTIL_WAVE_TW file

            file_found = False

            for calib_type in ["fpet", "une"]:
                if "UTIL_WAVE_TW" in self.file_dict:
                    for key in self.file_dict["UTIL_WAVE_TW"]:
                        if (
                            not file_found
                            and key.split("/")[-2] == f"util_wave_{calib_type}"
                        ):
                            file_name = key.split("/")[-2:]
                            print(
                                f"   - calib/{file_name[-2]}/{file_name[-1]} UTIL_WAVE_TW"
                            )
                            sof_open.write(f"{key} UTIL_WAVE_TW\n")
                            file_found = True

            # if "CAL_WAVE_TW" in self.file_dict:
            #     for key in self.file_dict["CAL_WAVE_TW"]:
            #         if not file_found:
            #             file_name = key.split("/")[-2:]
            #             print(f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_WAVE_TW")
            #             sof_open.write(f"{key} CAL_WAVE_TW\n")
            #             file_found = True
            #
            # if "CAL_FLAT_TW" in self.file_dict:
            #     for key in self.file_dict["CAL_FLAT_TW"]:
            #         if not file_found:
            #             file_name = key.split("/")[-2:]
            #             print(f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_FLAT_TW")
            #             sof_open.write(f"{key} CAL_FLAT_TW\n")
            #             file_found = True

            if not file_found:
                warnings.warn("Could not find file with TraceWave table.")

            # Find CAL_DETLIN_COEFFS file

            file_found = False

            if "CAL_DETLIN_COEFFS" in self.file_dict:
                for key in self.file_dict["CAL_DETLIN_COEFFS"]:
                    if not file_found:
                        file_name = key.split("/")[-2:]
                        print(
                            f"   - calib/{file_name[-2]}/{file_name[-1]} CAL_DETLIN_COEFFS"
                        )
                        sof_open.write(f"{key} CAL_DETLIN_COEFFS\n")
                        file_found = True

            if not file_found:
                warnings.warn("Could not find CAL_DETLIN_COEFFS.")

            sof_open.close()

            # Create EsoRex configuration file if not found

            self._create_config("cr2res_obs_nodding", "obs_nodding", verbose)

            # Run EsoRex

            print()

            config_file = self.config_folder / "obs_nodding.rc"

            esorex = [
                "esorex",
                f"--recipe-config={config_file}",
                f"--output-dir={output_dir}",
                "cr2res_obs_nodding",
                sof_file,
            ]

            if verbose:
                stdout = None
            else:
                stdout = subprocess.DEVNULL
                print("Running EsoRex...", end="", flush=True)

            subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

            if not verbose:
                print(" [DONE]\n")

            if correct_bad_pixels:
                for nod_pos in ["A", "B"]:
                    fits_file = (
                        output_dir / f"cr2res_obs_nodding_combined{nod_pos}.fits"
                    )

                    with fits.open(fits_file) as hdu_list:
                        # Iterate over 3 detectors
                        for det_idx in range(3):
                            # Read image with spectra
                            # Bad pixels are set to NaN
                            image = hdu_list[(det_idx * 2) + 1].data

                            # Create bad pixel mask
                            mask = np.zeros(image.shape)
                            mask[np.isnan(image)] = 1.0

                            # Overwrite the image
                            # Bad pixels are corrected by inpainting
                            hdu_list[
                                (det_idx * 2) + 1
                            ].data = inpaint.inpaint_biharmonic(image, mask)

                            bp_fraction = np.sum(np.isnan(image)) / np.size(image)
                            print(
                                f"Bad pixels in nod {nod_pos}, "
                                f"detector {det_idx+1}: "
                                f"{100.*bp_fraction:.1f}%"
                            )

                            # Read image with uncertainties
                            # Bad pixels are set to NaN
                            image = hdu_list[(det_idx * 2) + 2].data

                            # Create bad pixel mask
                            mask = np.zeros(image.shape)
                            mask[np.isnan(image)] = 1.0

                            # Overwrite the image
                            # Bad pixels are corrected by inpainting
                            hdu_list[
                                (det_idx * 2) + 2
                            ].data = inpaint.inpaint_biharmonic(image, mask)

                        hdu_list.writeto(fits_file, overwrite=True)

                print()

            spec_file = pathlib.Path(output_dir / "cr2res_obs_nodding_extractedA.fits")
            spec_file.rename(
                output_dir / f"cr2res_obs_nodding_extractedA_{count_exp:03d}.fits"
            )

            spec_file = pathlib.Path(output_dir / "cr2res_obs_nodding_extractedB.fits")
            spec_file.rename(
                output_dir / f"cr2res_obs_nodding_extractedB_{count_exp:03d}.fits"
            )

            spec_file = pathlib.Path(
                output_dir / "cr2res_obs_nodding_extracted_combined.fits"
            )
            spec_file.rename(
                output_dir
                / f"cr2res_obs_nodding_extracted_combined_{count_exp:03d}.fits"
            )

            spec_file = pathlib.Path(output_dir / "cr2res_obs_nodding_combinedA.fits")
            spec_file.rename(
                output_dir / f"cr2res_obs_nodding_combinedA_{count_exp:03d}.fits"
            )

            spec_file = pathlib.Path(output_dir / "cr2res_obs_nodding_combinedB.fits")
            spec_file.rename(
                output_dir / f"cr2res_obs_nodding_combinedB_{count_exp:03d}.fits"
            )

            spec_file = pathlib.Path(output_dir / "cr2res_obs_nodding_modelA.fits")
            spec_file.rename(
                output_dir / f"cr2res_obs_nodding_modelA_{count_exp:03d}.fits"
            )

            spec_file = pathlib.Path(output_dir / "cr2res_obs_nodding_modelB.fits")
            spec_file.rename(
                output_dir / f"cr2res_obs_nodding_modelB_{count_exp:03d}.fits"
            )

            spec_file = pathlib.Path(output_dir / "cr2res_obs_nodding_slitfuncA.fits")
            spec_file.rename(
                output_dir / f"cr2res_obs_nodding_slitfuncA_{count_exp:03d}.fits"
            )

            spec_file = pathlib.Path(output_dir / "cr2res_obs_nodding_slitfuncB.fits")
            spec_file.rename(
                output_dir / f"cr2res_obs_nodding_slitfuncB_{count_exp:03d}.fits"
            )

            spec_file = pathlib.Path(
                output_dir / "cr2res_obs_nodding_trace_wave_A.fits"
            )
            spec_file.rename(
                output_dir / f"cr2res_obs_nodding_trace_wave_A_{count_exp:03d}.fits"
            )

            spec_file = pathlib.Path(
                output_dir / "cr2res_obs_nodding_trace_wave_B.fits"
            )
            spec_file.rename(
                output_dir / f"cr2res_obs_nodding_trace_wave_B_{count_exp:03d}.fits"
            )

            # Update file dictionary with output files

            print(f"Output files for nod pair #{count_exp+1}/{indices.sum()//2}:")

            fits_file = (
                output_dir / f"cr2res_obs_nodding_combinedA_{count_exp:03d}.fits"
            )
            self._update_files("OBS_NODDING_COMBINEDA", str(fits_file))

            fits_file = (
                output_dir / f"cr2res_obs_nodding_combinedB_{count_exp:03d}.fits"
            )
            self._update_files("OBS_NODDING_COMBINEDB", str(fits_file))

            fits_file = (
                output_dir / f"cr2res_obs_nodding_extractedA_{count_exp:03d}.fits"
            )
            self._update_files("OBS_NODDING_EXTRACTA", str(fits_file))

            fits_file = (
                output_dir / f"cr2res_obs_nodding_extractedB_{count_exp:03d}.fits"
            )
            self._update_files("OBS_NODDING_EXTRACTB", str(fits_file))

            fits_file = output_dir / f"cr2res_obs_nodding_modelA_{count_exp:03d}.fits"
            self._update_files("OBS_NODDING_SLITMODELA", str(fits_file))

            fits_file = output_dir / f"cr2res_obs_nodding_modelB_{count_exp:03d}.fits"
            self._update_files("OBS_NODDING_SLITMODELB", str(fits_file))

            fits_file = (
                output_dir / f"cr2res_obs_nodding_slitfuncA_{count_exp:03d}.fits"
            )
            self._update_files("OBS_NODDING_SLITFUNCA", str(fits_file))

            fits_file = (
                output_dir / f"cr2res_obs_nodding_slitfuncB_{count_exp:03d}.fits"
            )
            self._update_files("OBS_NODDING_SLITFUNCB", str(fits_file))

            count_exp += 1

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def molecfit_input(self, nod_ab: str = "A") -> None:
        """
        Method for converting the extracted spectra into input files
        for `Molecfit`. The content of this method has been adapted
        from the ``cr2res_drs2molecfit.py`` code that is included
        with the EsoRex pipeline for CRIRES+
        (see https://www.eso.org/sci/software/pipelines/).

        Parameters
        ----------
        nod_ab : str
            Nod position of which the extracted spectra are plotted ("A" or "B").

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Create input for Molecfit")

        # Create output folder

        output_dir = self.calib_folder / "molecfit_input"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Read extracted spectrum

        fits_file = (
            f"{self.path}/product/cr2res_obs_nodding_extracted{nod_ab}_corr.fits"
        )

        if os.path.exists(fits_file):
            print(f"Input file: product/cr2res_obs_nodding_extracted{nod_ab}_corr.fits")

        else:
            fits_file = f"{self.path}/product/cr2res_obs_nodding_extracted{nod_ab}.fits"
            print(f"Input file: product/cr2res_obs_nodding_extracted{nod_ab}.fits")

        print("\nOutput files:")

        with fits.open(fits_file) as hdu_list:
            # Copy header from the primary HDU
            hdu_out = fits.HDUList([hdu_list[0]])
            hdu_win = fits.HDUList([hdu_list[0]])
            hdu_atm = fits.HDUList([hdu_list[0]])
            hdu_con = fits.HDUList([hdu_list[0]])
            hdu_cor = fits.HDUList([hdu_list[0]])

            # Initialize empty lists
            wmin = []
            wmax = []
            map_chip = []
            map_ext = [0]
            wlc_fit = []

            # Initialize counter for order/detector
            count = 1

            # Loop over 3 detectors
            for i_det in range(3):
                # Get detector data
                data = hdu_list[f"CHIP{i_det+1}.INT1"].data

                # Get all spectral orders
                spec_orders = np.sort([i[:5] for i in data.dtype.names if "WL" in i])

                # Loop over spectral orders
                for item in spec_orders:
                    # Extract WL, SPEC, and ERR for given order/detector
                    wavel = hdu_list[f"CHIP{i_det+1}.INT1"].data[item + "_WL"]
                    spec = hdu_list[f"CHIP{i_det+1}.INT1"].data[item + "_SPEC"]
                    err = hdu_list[f"CHIP{i_det+1}.INT1"].data[item + "_ERR"]

                    spec = np.nan_to_num(spec)
                    err = np.nan_to_num(err)

                    # Convert wavelength from nm to um
                    wavel *= 1e-3

                    # Create a FITS columns for WAVE, SPEC, and ERR
                    col1 = fits.Column(name="WAVE", format="D", array=wavel)
                    col2 = fits.Column(name="SPEC", format="D", array=spec)
                    col3 = fits.Column(name="ERR", format="D", array=err)

                    # Create FITS table with WAVE, SPEC, and ERR
                    table_hdu = fits.BinTableHDU.from_columns([col1, col2, col3])

                    # Append table HDU to output of HDU list
                    hdu_out.append(table_hdu)

                    # Minimum and maximum wavelengths for a given order
                    wmin.append(np.min(wavel))
                    wmax.append(np.max(wavel))

                    # Mapping for order/detector
                    map_chip.append(count)
                    map_ext.append(count)
                    wlc_fit.append(1)

                    count += 1

            # Create FITS file with SCIENCE
            print(f"   - calib/molecfit_input/SCIENCE_{nod_ab}.fits")
            hdu_out.writeto(output_dir / f"SCIENCE_{nod_ab}.fits", overwrite=True)

            # Create FITS file with WAVE_INCLUDE
            col_wmin = fits.Column(name="LOWER_LIMIT", format="D", array=wmin)
            col_wmax = fits.Column(name="UPPER_LIMIT", format="D", array=wmax)
            col_map = fits.Column(name="MAPPED_TO_CHIP", format="I", array=map_chip)
            col_wlc = fits.Column(name="WLC_FIT_FLAG", format="I", array=wlc_fit)
            col_cont = fits.Column(name="CONT_FIT_FLAG", format="I", array=wlc_fit)
            columns = [col_wmin, col_wmax, col_map, col_wlc, col_cont]
            table_hdu = fits.BinTableHDU.from_columns(columns)
            hdu_win.append(table_hdu)
            print(f"   - calib/molecfit_input/WAVE_INCLUDE_{nod_ab}.fits")
            hdu_win.writeto(output_dir / f"WAVE_INCLUDE_{nod_ab}.fits", overwrite=True)

            # Create FITS file with MAPPING_ATMOSPHERIC
            name = "ATM_PARAMETERS_EXT"
            col_atm = fits.Column(name=name, format="K", array=map_ext)
            table_hdu = fits.BinTableHDU.from_columns([col_atm])
            hdu_atm.append(table_hdu)
            print(f"   - calib/molecfit_input/MAPPING_ATMOSPHERIC_{nod_ab}.fits")
            fits_file = output_dir / f"MAPPING_ATMOSPHERIC_{nod_ab}.fits"
            hdu_atm.writeto(fits_file, overwrite=True)

            # Create FITS file with MAPPING_CONVOLVE
            name = "LBLRTM_RESULTS_EXT"
            col_conv = fits.Column(name=name, format="K", array=map_ext)
            table_hdu = fits.BinTableHDU.from_columns([col_conv])
            hdu_con.append(table_hdu)
            print(f"   - calib/molecfit_input/MAPPING_CONVOLVE_{nod_ab}.fits")
            fits_file = output_dir / f"MAPPING_CONVOLVE_{nod_ab}.fits"
            hdu_con.writeto(fits_file, overwrite=True)

            # Create FITS file with MAPPING_CORRECT
            name = "TELLURIC_CORR_EXT"
            col_corr = fits.Column(name=name, format="K", array=map_ext)
            table_hdu = fits.BinTableHDU.from_columns([col_corr])
            hdu_cor.append(table_hdu)
            print(f"   - calib/molecfit_input/MAPPING_CORRECT_{nod_ab}.fits")
            fits_file = output_dir / f"MAPPING_CORRECT_{nod_ab}.fits"
            hdu_cor.writeto(fits_file, overwrite=True)

    @typechecked
    def molecfit_model(self, nod_ab: str = "A", verbose: bool = True) -> None:
        """
        Method for running ``molecfit_model``.

        Parameters
        ----------
        nod_ab : str
            Nod position of which the extracted spectra are plotted
            ("A" or "B").
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Fit telluric model", recipe_name="molecfit_model")

        # Create output folder

        input_dir = self.calib_folder / "molecfit_input"
        output_dir = self.calib_folder / "molecfit_model"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(output_dir / "files.sof")

        with open(sof_file, "w", encoding="utf-8") as sof_open:
            print(f"   - calib/molecfit_input/SCIENCE_{nod_ab}.fits SCIENCE")
            sof_open.write(f"{input_dir / f'SCIENCE_{nod_ab}.fits'} SCIENCE\n")

            print(f"   - calib/molecfit_input/WAVE_INCLUDE_{nod_ab}.fits WAVE_INCLUDE")
            sof_open.write(
                f"{input_dir / f'WAVE_INCLUDE_{nod_ab}.fits'} WAVE_INCLUDE\n"
            )

        # Create EsoRex configuration file if not found

        self._create_config("molecfit_model", "molecfit_model", verbose)

        # Read molecules from config file

        config_file = self.config_folder / "molecfit_model.rc"

        with open(config_file, "r", encoding="utf-8") as open_config:
            config_text = open_config.readlines()

        print("\nSelected molecules:")
        for line_item in config_text:
            if line_item[:10] == "LIST_MOLEC":
                for mol_item in line_item[11:].split(","):
                    print(f"   - {mol_item}")

        # Run EsoRex

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "molecfit_model",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def molecfit_calctrans(self, nod_ab: str = "A", verbose: bool = True) -> None:
        """
        Method for running ``molecfit_calctrans``.

        Parameters
        ----------
        nod_ab : str
            Nod position of which the extracted spectra are plotted
            ("A" or "B").
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section(
            "Generate telluric spectrum", recipe_name="molecfit_calctrans"
        )

        # Create output folder

        input_dir = self.calib_folder / "molecfit_input"
        model_dir = self.calib_folder / "molecfit_model"
        output_dir = self.calib_folder / "molecfit_calctrans"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(output_dir / "files.sof")

        with open(sof_file, "w", encoding="utf-8") as sof_open:
            print(f"   - calib/molecfit_input/SCIENCE_{nod_ab}.fits SCIENCE")
            sof_open.write(f"{input_dir / f'SCIENCE_{nod_ab}.fits'} SCIENCE\n")

            print(
                f"   - calib/molecfit_input/MAPPING_ATMOSPHERIC_{nod_ab}.fits MAPPING_ATMOSPHERIC"
            )
            sof_open.write(
                f"{input_dir / f'MAPPING_ATMOSPHERIC_{nod_ab}.fits'} MAPPING_ATMOSPHERIC\n"
            )

            print(
                f"   - calib/molecfit_input/MAPPING_CONVOLVE_{nod_ab}.fits MAPPING_CONVOLVE"
            )
            sof_open.write(
                f"{input_dir / f'MAPPING_CONVOLVE_{nod_ab}.fits'} MAPPING_CONVOLVE\n"
            )

            print("   - calib/molecfit_model/ATM_PARAMETERS.fits ATM_PARAMETERS")
            sof_open.write(f"{model_dir / 'ATM_PARAMETERS.fits'} ATM_PARAMETERS\n")

            print("   - calib/molecfit_model/MODEL_MOLECULES.fits MODEL_MOLECULES")
            sof_open.write(f"{model_dir / 'MODEL_MOLECULES.fits'} MODEL_MOLECULES\n")

            print(
                "   - calib/molecfit_model/BEST_FIT_PARAMETERS.fits BEST_FIT_PARAMETERS"
            )
            sof_open.write(
                f"{model_dir / 'BEST_FIT_PARAMETERS.fits'} BEST_FIT_PARAMETERS\n"
            )

        # Create EsoRex configuration file if not found

        self._create_config("molecfit_calctrans", "molecfit_calctrans", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "molecfit_calctrans.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "molecfit_calctrans",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def molecfit_correct(self, nod_ab: str = "A", verbose: bool = True) -> None:
        """
        Method for running ``molecfit_correct``.

        Parameters
        ----------
        nod_ab : str
            Nod position of which the extracted spectra are plotted
            ("A" or "B").
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Apply telluric correction", recipe_name="molecfit_correct")

        # Create output folder

        input_dir = self.calib_folder / "molecfit_input"
        calctrans_dir = self.calib_folder / "molecfit_calctrans"
        output_dir = self.calib_folder / "molecfit_correct"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(output_dir / "files.sof")

        with open(sof_file, "w", encoding="utf-8") as sof_open:
            print(f"   - calib/molecfit_input/SCIENCE_{nod_ab}.fits SCIENCE")
            sof_open.write(f"{input_dir / f'SCIENCE_{nod_ab}.fits'} SCIENCE\n")

            print(
                f"   - calib/molecfit_input/MAPPING_CORRECT_{nod_ab}.fits MAPPING_CORRECT"
            )
            sof_open.write(
                f"{input_dir / f'MAPPING_CORRECT_{nod_ab}.fits'} MAPPING_CORRECT\n"
            )

            print("   - calib/molecfit_calctrans/TELLURIC_CORR.fits TELLURIC_CORR")
            sof_open.write(f"{calctrans_dir / 'TELLURIC_CORR.fits'} TELLURIC_CORR\n")

        # Create EsoRex configuration file if not found

        self._create_config("molecfit_correct", "molecfit_correct", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "molecfit_correct.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={output_dir}",
            "molecfit_correct",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def xcor_wavelength_solution(
        self,
        spec: np.ndarray,
        wavel: np.ndarray,
        telluric_template: np.ndarray,
        accuracy: float = 0.01,
        window_length: int = 201,
    ) -> tuple([np.ndarray, float, float]):

        template_interp = interpolate.interp1d(
            telluric_template[:, 0],
            telluric_template[:, 1],
            kind="linear",
            bounds_error=True,
        )

        # Remove continuum and nans of spectra.
        # The continuum is estimated by smoothing the
        # spectrum with a 2nd order Savitzky-Golay filter
        nans = np.isnan(spec)
        continuum = signal.savgol_filter(
            spec[~nans], window_length=window_length, polyorder=2, mode="interp"
        )

        # Add 1 to continuum for bins where there is no flux
        spec_flat = spec[~nans] / (continuum + 1) - 1.0

        # Don't use the edges as that sometimes gives problems
        spec_flat = spec_flat[20:-20]
        used_wavel = wavel[~nans][20:-20]

        # Prepare cross-correlation grid
        dlam = (wavel[-1] - np.mean(wavel)) / 2
        da = accuracy / dlam
        N_a = int(np.ceil(0.1 / da) // 2 * 2 + 1)
        N_b = int(np.ceil(0.6 / accuracy) // 2 * 2 + 1)

        a_grid = np.linspace(0.95, 1.05, N_a)[:, np.newaxis, np.newaxis]
        b_grid = np.linspace(-0.3, 0.3, N_b)[np.newaxis, :, np.newaxis]
        mean_wavel = np.mean(wavel)
        wl_matrix = (
            a_grid * (used_wavel[np.newaxis, np.newaxis, :] - mean_wavel)
            + mean_wavel
            + b_grid
        )
        template = template_interp(wl_matrix) - 1.0

        # Calculate the cross-correlation
        # between data and template
        cross_corr = template.dot(spec_flat)

        # Find optimal wavelength solution
        opt_idx = np.unravel_index(np.argmax(cross_corr), cross_corr.shape)
        opt_a = a_grid[opt_idx[0], 0, 0]
        opt_b = b_grid[0, opt_idx[1], 0]
        return cross_corr, opt_a, opt_b

    @typechecked
    def correct_wavelengths(
        self,
        nod_ab: str = "A",
        accuracy: float = 0.01,
        window_length: int = 201,
        minimum_strength: float = 0.005,
        create_plots: bool = False,
    ) -> None:
        """
        Method for correcting the wavelength solution with a linear
        function and maximizing the correlation with the telluric
        model spectrum from SkyCalc, obtained with
        :func:`~pycrires.pipeline.Pipeline.run_skycalc`.

        Parameters
        ----------
        nod_ab : str
            Nod position with the spectra of which the wavelength
            solution will be corrected.
        accuracy : float
            Desired accuracy in nm of the wavelength solution.
            This will be used to generate the grid on which the
            correlation with a telluric spectrum is calculated
            (default: 0.01 nm).
        window_length : int
            Width of the kernel (in number of pixels) that is used
            for estimating the continuum by smoothing with the 2nd
            order Savitzky-Golay filter from
            ``scipy.signal.savgol_filter`` (default: 201).
        minimum_strength : float
            Minimum standard deviation of the telluric spectrum in
            the wavelength range of the spectral order that is used
            as threshold for applying the wavelength correction. If
            there are not enough features (i.e. the standard
            deviation is smaller than ``minimum_strength``), the
            original wavelength solution will be saved (default:
            0.005).
        create_plots : bool
            Create plots with the correlation maps (default: False).

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Correct wavelength solution")

        # Create output folder

        output_dir = self.product_folder / "correct_wavelengths"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Model file: run_skycalc/transm_spec.dat")
        print(f"Spectrum file: product/cr2res_obs_nodding_extracted{nod_ab}.fits")

        # Read telluric model

        print("Reading telluric model spectrum...", end="", flush=True)

        transm_spec = np.loadtxt(self.calib_folder / "run_skycalc/transm_spec.dat")
        transm_interp = interpolate.interp1d(
            transm_spec[:, 0], transm_spec[:, 1], kind="linear", bounds_error=True
        )

        print(" [DONE]")

        # Read extracted spectra

        fits_files = list(self.file_dict[f"OBS_NODDING_EXTRACT{nod_ab}"].keys())

        for fits_file in fits_files:

            print(f"\nReading spectra from {fits_file}...", end="", flush=True)

            hdu_list = fits.open(fits_file)

            print(" [DONE]")

            if create_plots:
                fig, axes = plt.subplots(
                    7, 3, figsize=(9, 15), sharex=True, sharey=True
                )

            for i_det in range(3):
                # Get detector data
                data = hdu_list[f"CHIP{i_det+1}.INT1"].data

                # Get all spectral orders
                spec_orders = np.sort([i[:5] for i in data.dtype.names if "WL" in i])

                print(f"\nCorrecting wavelength solution of detector {i_det+1}:")

                # Loop over spectral orders
                for order, spec_name in enumerate(spec_orders):
                    # Extract WL, SPEC, and ERR for given order/detector
                    wavel = hdu_list[f"CHIP{i_det+1}.INT1"].data[spec_name + "_WL"]
                    spec = hdu_list[f"CHIP{i_det+1}.INT1"].data[spec_name + "_SPEC"]
                    # err = hdu_list[f"CHIP{i_det+1}.INT1"].data[spec_name + "_ERR"]

                    # Check if there are enough telluric
                    # features in this wavelength range
                    wl_mask = (transm_spec[:, 0] > np.min(wavel)) * (
                        transm_spec[:, 0] < np.max(wavel)
                    )
                    template_std = np.std(transm_spec[wl_mask, 1])
                    if template_std > minimum_strength:
                        # Calculate the cross-correlation
                        # between data and template
                        cross_corr, opt_a, opt_b = self.xcor_wavelength_solution(
                            spec, wavel, transm_spec, accuracy, window_length
                        )

                        # Plot correlation map
                        if create_plots:
                            plt.sca(axes[order, i_det])
                            plt.title(f"Detector {i_det+1}, order {order}")
                            plt.imshow(
                                cross_corr,
                                extent=[-0.5, 0.5, 0.9, 1.1],
                                origin="lower",
                                aspect="auto",
                            )
                            plt.colorbar()
                            plt.axhline(opt_a, ls="--", color="white")
                            plt.axvline(opt_b, ls="--", color="white")

                    else:
                        warnings.warn(
                            "Not enough telluric features to "
                            "correct wavelength for detector "
                            f"{i_det} and order {spec_name}"
                        )
                        opt_a = 1.0
                        opt_b = 0.0

                        if create_plots:
                            plt.sca(axes[order, i_det])
                            plt.axis("off")

                    print(
                        f"   - {spec_name} -> lambda = {opt_b:.4f} "
                        f"+ {opt_a:.4f} * lambda'"
                    )

                    mean_wavel = np.mean(wavel)
                    hdu_list[f"CHIP{i_det+1}.INT1"].data[spec_name + "_WL"] = (
                        opt_a * (wavel - mean_wavel) + mean_wavel + opt_b
                    )

            # Write the corrected spectra to a new FITS file

            out_file = output_dir / (pathlib.Path(fits_file).stem + "_corr.fits")
            print(f"\nStoring corrected spectra: {out_file}")

            hdu_list.writeto(out_file, overwrite=True)

            # Save the correlation plots
            if create_plots:
                fig.add_subplot(111, frame_on=False)
                plt.tick_params(labelcolor="none", bottom=False, left=False)
                plt.ylabel("Slope", fontsize=16)
                plt.xlabel("Offset (nm)", fontsize=16)
                out_label = fits_file[-10:-5]
                plt.tight_layout()
                plt.savefig(f"{output_dir}/correlation_map_{out_label}.png")

    @typechecked
    def correct_wavelengths_2d(
        self,
        nod_ab: str = "A",
        accuracy: float = 0.01,
        window_length: int = 201,
        minimum_strength: float = 0.005,
        sum_over_spatial_dim: bool = True,
        input_folder="fit_gaussian",
    ) -> None:
        """
        Method for correcting the wavelength solution with a linear
        function and maximizing the correlation with the telluric
        model spectrum from SkyCalc, obtained with
        :func:`~pycrires.pipeline.Pipeline.run_skycalc`. This
        function can be applied to 2D extracted spectra, when one
        wants to keep the spatial information.

        Parameters
        ----------
        nod_ab : str
            Nod position with the spectra of which the wavelength
            solution will be corrected.
        accuracy : float
            Desired accuracy in nm of the wavelength solution.
            This will be used to generate the grid on which the
            correlation with a telluric spectrum is calculated
            (default: 0.01 nm).
        window_length : int
            Width of the kernel (in number of pixels) that is used
            for estimating the continuum by smoothing with the 2nd
            order Savitzky-Golay filter from
            ``scipy.signal.savgol_filter`` (default: 201).
        minimum_strength : float
            Minimum standard deviation of the telluric spectrum in
            the wavelength range of the spectral order that is used
            as threshold for applying the wavelength correction. If
            there are not enough features (i.e. the standard
            deviation is smaller than ``minimum_strength``), the
            original wavelength solution will be saved (default:
            0.005).
        sum_over_spatial_dim: bool
            If True, the wavelength correction will be calculated
            using the summed spectra over the spatial direction,
            improving the S/N of the spectra. However, this may
            result in slight errors in the wavelength solution
            off-axis.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Correct wavelength solution")

        # Create output folder

        # output_dir = self.product_folder / "correct_wavelengths_2d"
        #
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        print("Model file: run_skycalc/transm_spec.dat")

        # Read telluric model

        print("Reading telluric model spectrum...", end="", flush=True)

        transm_spec = np.loadtxt(self.calib_folder / "run_skycalc/transm_spec.dat")

        print(" [DONE]")

        # Read extracted spectra

        if input_folder == "custom_extract_2d":
            fits_files = list(self.file_dict[f"CUSTOM_EXTRACT_2D_{nod_ab}"].keys())

        elif input_folder == "util_extract_2d":
            fits_files = list(self.file_dict[f"UTIL_EXTRACT_2D_{nod_ab}"].keys())

        elif input_folder == "fit_gaussian":
            fits_files = list(self.file_dict[f"FIT_GAUSSIAN_2D_{nod_ab}"].keys())

        for fits_file in fits_files:
            print(f"\nReading spectra from {fits_file}...", end="", flush=True)

            hdu_list = fits.open(fits_file)
            corrected_wavel = hdu_list["WAVE"].data

            print(" [DONE]")

            for det_idx in range(3):
                # Get detector data
                print("\nCorrecting wavelength solution " f"of detector {det_idx+1}:")

                # Loop over spectral orders
                for order_idx in np.arange(7):
                    # Extract WL and SPEC for given order/detector
                    wavel_2d = hdu_list["WAVE"].data[det_idx, order_idx]
                    spec_2d = hdu_list["SPEC"].data[det_idx, order_idx]

                    # If we sum over the spatial dimension to boost SNR, do so
                    if sum_over_spatial_dim:
                        wavel_list = [np.nanmean(wavel_2d, axis=0)]
                        spec_list = [np.nansum(spec_2d, axis=0)]

                    else:
                        wavel_list = np.copy(wavel_2d)
                        spec_list = np.copy(spec_2d)

                    # Check if there are enough telluric
                    # features in this wavelength range

                    wl_mask = (transm_spec[:, 0] > np.min(wavel_2d)) * (
                        transm_spec[:, 0] < np.max(wavel_2d)
                    )

                    template_std = np.std(transm_spec[wl_mask, 1])

                    if not template_std > minimum_strength:
                        warnings.warn(
                            "Not enough telluric features to "
                            f"correct wavelength for detector "
                            f"{det_idx} and order {order_idx}, "
                            "using EsoRex wavelength solution"
                        )

                        corrected_wavel[det_idx, order_idx, :] = wavel_2d

                    else:
                        for row, (spec, wavel) in enumerate(zip(spec_list, wavel_list)):
                            # Calculate wavelength solution using cross-correlation

                            cross_corr, opt_a, opt_b = self.xcor_wavelength_solution(
                                spec, wavel, transm_spec, accuracy, window_length
                            )

                            print(
                                f"   - Detector {det_idx+1}, "
                                f"Order {order_idx}, Row {row} "
                                f"-> lambda = {opt_b:.4f} "
                                f"+ {opt_a:.4f} * lambda'"
                            )

                            mean_wavel = np.mean(wavel)

                            # Save new wavelength solution

                            if sum_over_spatial_dim:
                                corrected_wavel[det_idx, order_idx, :] = (
                                    opt_a * (wavel - mean_wavel) + mean_wavel + opt_b
                                )

                            else:
                                corrected_wavel[det_idx, order_idx, row] = (
                                    opt_a * (wavel - mean_wavel) + mean_wavel + opt_b
                                )

            # Add corrected wavelengths to a new HDU
            # called CORR_WAVE in existing FITS files

            if "CORR_WAVE" in hdu_list:
                hdu_list.pop("CORR_WAVE")

            hdu_list.insert(4, fits.ImageHDU(corrected_wavel, name="CORR_WAVE"))
            hdu_list.writeto(fits_file, overwrite=True)

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def util_extract_2d(
        self,
        nod_ab: str = "A",
        verbose: bool = True,
        extraction_length: float = 0.059,
        spatial_oversampling: float = 1.0,
        use_corr_wavel: bool = True,
    ) -> None:
        """
        Method for extracting spectra from the products of
        :func:`~pycrires.pipeline.Pipeline.obs_nodding`, while
        retaining the spatial information. It is important
        to run :func:`~pycrires.pipeline.Pipeline.obs_nodding` before
        :func:`~pycrires.pipeline.Pipeline.util_extract_2d`. The
        2D spectra are extracted by running the ``cr2res_util_extract``
        recipe over a range of slit fractions.

        Parameters
        ----------
        nod_ab : str
            Nod position with the spectra of which the wavelength
            solution will be corrected.
        verbose : bool
            Print output produced by ``esorex``.
        extraction_length : float
            Spatial extent (arcsec) over which to extract the spectrum.
            The default value of 0.059 arcsec is the pixel scale of the
            CRIRES detectors.
        spatial_oversampling : float
            Oversampling factor for the extraction along the slit
            length. For example, with ``spatial_oversampling=2``, the
            end result will have twice as many spatial pixels as the
            original images.
        use_corr_wavel : bool
            Use the wavelength solution obtained with
            :func:`~pycrires.pipeline.Pipeline.correct_wavelengths`
            when set to ``True``. Otherwise, the original wavelength
            solution is used that is determined by the EsoRex recipes.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Extract 2D spectra", recipe_name="cr2res_util_extract")

        # Create output folder

        output_dir = self.product_folder / "util_extract_2d"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_util_extract", "util_extract_2d", verbose)

        # List with FITS files that will be processed

        fits_files = sorted(
            list(self.file_dict[f"OBS_NODDING_COMBINED{nod_ab}"].keys())
        )

        # Iterate over exposures
        for count_exp, fits_file in enumerate(fits_files):
            if count_exp > 0:
                print()

            print(
                f"Creating SOF file for exposure " f"#{count_exp+1}/{len(fits_files)}:"
            )

            # Prepare SOF file
            sof_file = pathlib.Path(
                self.product_folder / "util_extract_2d" / f"files_{count_exp:03d}.sof"
            )
            with open(sof_file, "w", encoding="utf-8") as sof_open:
                sof_open.write(f"{fits_file} OBS_NODDING_OTHER\n")
                file_name = fits_file.split("/")[-2:]
                print(f"   - product/{file_name[-2]}/{file_name[-1]} OBS_NODDING_OTHER")

                tw_file = (
                    self.product_folder
                    / f"obs_nodding/cr2res_obs_nodding_trace_wave_{nod_ab}_{count_exp:03d}.fits"
                )
                sof_open.write(f"{tw_file} UTIL_WAVE_TW\n")
                file_name = str(tw_file).split("/")[-2:]
                print(f"   - product/{file_name[-2]}/{file_name[-1]} UTIL_WAVE_TW\n")

            # Get data on size of the slit on the detector
            tw_data = fits.open(tw_file)[1].data
            slit_range = tw_data["SlitFraction"][0]
            num_orders = tw_data["Order"].size
            tot_slit_fraction = slit_range[-1] - slit_range[0]
            full_slit_length = 10  # (arcsec)

            # Calculate the spatial size for the extraction
            # and the number of spatial points
            extraction_fraction = extraction_length / full_slit_length
            n_points = spatial_oversampling * tot_slit_fraction / extraction_fraction

            # Round n_points up to the next odd integer
            n_points = int(np.ceil(n_points) // 2 * 2 + 1)

            # Calculate slit fraction centers to use for util_extract
            extraction_centers = np.linspace(
                0, tot_slit_fraction - extraction_fraction, n_points
            )

            # Make sure the middle value is centered around the star
            extraction_centers = (
                extraction_centers - np.median(extraction_centers)
            ) + slit_range[1]

            # Allocate arrays to save the results
            flux_2d = np.zeros((3, num_orders, n_points, 2048))
            errors_2d = np.zeros((3, num_orders, n_points, 2048))
            wavelengths = np.zeros((3, num_orders, n_points, 2048))

            for pos_idx, center in enumerate(extraction_centers):
                lower_lim = center - extraction_fraction / 2
                upper_lim = center + extraction_fraction / 2

                print(
                    f"--> Extracting spectrum between slit "
                    f"fractions: {lower_lim:.4f} - {upper_lim:.4f}"
                )

                # Run EsoRex

                config_file = self.config_folder / "util_extract_2d.rc"

                esorex = [
                    "esorex",
                    f"--recipe-config={config_file}",
                    f"--output-dir={self.product_folder / 'util_extract_2d'}",
                    "cr2res_util_extract",
                    f"--slit_frac={lower_lim},{upper_lim}",
                    sof_file,
                ]

                if verbose:
                    stdout = None
                else:
                    stdout = subprocess.DEVNULL
                    print("Running EsoRex...", end="", flush=True)

                subprocess.run(esorex, cwd=output_dir, stdout=stdout, check=True)

                if not verbose:
                    print(" [DONE]\n")

                # Load 1D result and store in array

                fits_out = (
                    self.product_folder
                    / "util_extract_2d"
                    / f"cr2res_obs_nodding_combined{nod_ab}_"
                    f"{count_exp:03d}_extr1D.fits"
                )

                hdu_list = fits.open(fits_out)

                for det_idx in np.arange(3):
                    for order_idx, order_item in enumerate(tw_data["Order"]):
                        flux_2d[det_idx, order_idx, pos_idx, :] = hdu_list[
                            det_idx + 1
                        ].data[f"{order_item:02d}_01_SPEC"]

                        errors_2d[det_idx, order_idx, pos_idx, :] = hdu_list[
                            det_idx + 1
                        ].data[f"{order_item:02d}_01_ERR"]

                        # Check wether to use the corrected wavelength
                        # solution or the original wavelength solution
                        # that was determined by the EsoRex recipes
                        if use_corr_wavel:
                            wave_file = (
                                self.product_folder
                                / "correct_wavelengths"
                                / "cr2res_obs_nodding_extracted"
                                f"{nod_ab}_{count_exp:03d}_corr.fits"
                            )

                            with fits.open(wave_file) as wave_hdu:
                                wavelengths[det_idx, order_idx, pos_idx, :] = wave_hdu[
                                    det_idx + 1
                                ].data[f"{order_item:02d}_01_WL"]

                        else:
                            wavelengths[det_idx, order_idx, pos_idx, :] = hdu_list[
                                det_idx + 1
                            ].data[f"{order_item:02d}_01_WL"]

                hdu_list.close()

            # Save 2D results

            fits_out = (
                self.product_folder
                / "util_extract_2d/"
                / f"cr2res_combined{nod_ab}_{count_exp:03d}_extr2d.fits"
            )

            result_hdu = fits.HDUList(fits.PrimaryHDU())
            result_hdu.append(fits.ImageHDU(flux_2d, name="SPEC"))
            result_hdu.append(fits.ImageHDU(errors_2d, name="ERR"))
            result_hdu.append(fits.ImageHDU(wavelengths, name="WAVE"))
            result_hdu.writeto(fits_out, overwrite=True)

            print(f"--> Done! Results written to {fits_out.stem}")

            # Remove redundant output files

            file_ext = ["extr1D", "extrModel", "extrSlitFu"]

            for file_item in file_ext:
                file_name = (
                    self.product_folder
                    / "util_extract_2d"
                    / f"cr2res_obs_nodding_combined{nod_ab}_"
                    f"{count_exp:03d}_{file_item}.fits"
                )

                os.remove(file_name)

    @typechecked
    def custom_extract_2d(
        self,
        nod_ab: str = "A",
        spatial_sampling: float = 0.059,
        max_separation: float = 2.0,
    ) -> None:
        """
        Method for extracting spectra from the products of
        :func:`~pycrires.pipeline.Pipeline.obs_nodding`, while
        retaining the spatial information. It is important to
        run :func:`~pycrires.pipeline.Pipeline.obs_nodding` before
        :func:`~pycrires.pipeline.Pipeline.custom_extract_2d`. The
        2D spectra are extracted using a custom pipeline that uses
        the trace data from the EsoRex pipeline.

        Parameters
        ----------
        nod_ab : str
            Nod position with the spectra of which the wavelength
            solution will be corrected.
        spatial_sampling : float
            Spatial interval (arcsec) over which to extract the spectrum.
            The default value of 0.059 arcsec is the pixel scale of the
            CRIRES detectors.
        max_separation : float
            Spatial extent (arcsec) over which to extract the spectrum.
            The default value of 2 arcsec is the size of the slit fraction
            typically used for nodding observations with CRIRES.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Extract 2D spectra")

        # Create output folder

        output_dir = self.product_folder / "custom_extract_2d"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List with FITS files that will be processed

        fits_files = sorted(
            list(self.file_dict[f"OBS_NODDING_COMBINED{nod_ab}"].keys())
        )

        out_files = []

        # Iterate over exposures
        for count_exp, fits_file in enumerate(fits_files):
            hdu_list = fits.open(fits_file)

            tw_file = (
                self.product_folder
                / f"obs_nodding/cr2res_obs_nodding_trace_wave_{nod_ab}_{count_exp:03d}.fits"
            )

            trace_hdu = fits.open(tw_file)
            tw_data = fits.open(tw_file)[1].data

            # Get data on size of the slit on the detector

            tw_data = fits.open(tw_file)[1].data
            slit_range = tw_data["SlitFraction"][0]
            num_orders = tw_data["Order"].size
            slit_fraction = slit_range[-1] - slit_range[0]
            full_slit_length = 10  # (arcsec)

            extraction_ratio = 2 * max_separation / (slit_fraction * full_slit_length)

            if extraction_ratio > 1:
                warnings.warn(
                    "Extracting a larger area than the slit "
                    "fraction used in this nodding position"
                )

            # Calculate the spatial size for the extraction
            # and the number of spatial points
            n_points = 2.0 * max_separation / spatial_sampling

            # Round n_points up to the next odd integer
            n_points = int(np.ceil(n_points) // 2 * 2 + 1)

            num_orders = tw_data["Order"].size

            # Make array for saving the results
            flux_2d = np.zeros((3, num_orders, n_points, 2048))
            errors_2d = np.zeros((3, num_orders, n_points, 2048))
            wavelengths = np.zeros((3, num_orders, n_points, 2048))

            for det_idx in np.arange(3):
                image = hdu_list[2 * det_idx + 1].data
                errors = hdu_list[2 * det_idx + 2].data
                trace_data = trace_hdu[det_idx + 1].data

                num_orders = tw_data["Order"].size

                xs = np.arange(1, image.shape[1] + 1)
                Y = np.arange(1, image.shape[1] + 1)

                for order_idx in range(num_orders):
                    cent_idx = trace_data["All"][order_idx, 0]
                    upper_idx = trace_data["Upper"][order_idx, 0]
                    lower_idx = trace_data["Lower"][order_idx, 0]

                    slit_fraction_dy = upper_idx - lower_idx
                    dy = extraction_ratio * slit_fraction_dy

                    # Make symmetric sampling around star position
                    y0s = np.linspace(cent_idx - dy / 2, cent_idx + dy / 2, n_points)

                    # Loop over spatial positions to extract
                    for pos_idx, y0 in enumerate(y0s):
                        print_msg = (
                            f"\rExtracting line {pos_idx}/{len(y0s)} "
                            f"of order {order_idx+1}/{num_orders} "
                            f"of detector {det_idx+1}/3 "
                            f"of exposure {count_exp+1}/"
                            f"{len(fits_files)}..."
                        )

                        print(print_msg, end="", flush=True)

                        # Calculate y coordinates of the trace
                        # using the central, upper and lower trace coordinates
                        y_mid = (
                            y0
                            + trace_data["All"][order_idx, 1] * xs
                            + trace_data["All"][order_idx, 2] * xs**2
                        )

                        if y0 > cent_idx:
                            y_up = (
                                y0
                                + trace_data["Upper"][order_idx, 1] * xs
                                + trace_data["Upper"][order_idx, 2] * xs**2
                            )

                            frac = (y0 - cent_idx) / y0
                            ys = frac * y_up + (1 - frac) * y_mid

                        else:
                            y_low = (
                                y0
                                + trace_data["Lower"][order_idx, 1] * xs
                                + trace_data["Lower"][order_idx, 2] * xs**2
                            )

                            frac = (cent_idx - y0) / y0
                            ys = frac * y_low + (1 - frac) * y_mid

                        # Interpolate the spectrum and error for the new coordinates
                        new_spec = [
                            interpolate.interp1d(
                                Y, image[:, x - 1], fill_value=0, bounds_error=False
                            )(y)
                            for (x, y) in zip(xs, ys)
                        ]

                        new_err = [
                            interpolate.interp1d(
                                Y, errors[:, x - 1], fill_value=0, bounds_error=False
                            )(y)
                            for (x, y) in zip(xs, ys)
                        ]

                        # Calculate the wavelength solution from the EsoRex data
                        new_waves = (
                            trace_data["Wavelength"][order_idx, 0]
                            + trace_data["Wavelength"][order_idx, 1] * xs
                            + trace_data["Wavelength"][order_idx, 2] * xs**2
                        )

                        # Save the results in the arrays

                        flux_2d[det_idx, order_idx, pos_idx, :] = new_spec
                        errors_2d[det_idx, order_idx, pos_idx, :] = new_err
                        wavelengths[det_idx, order_idx, pos_idx, :] = new_waves

                        print("\r" + len(print_msg) * " ", end="", flush=True)

            hdu_list.close()

            # Save 2D results

            fits_out = (
                self.product_folder
                / "custom_extract_2d/"
                / f"cr2res_combined{nod_ab}_{count_exp:03d}_extr2d.fits"
            )

            result_hdu = fits.HDUList(fits.PrimaryHDU())
            result_hdu.append(fits.ImageHDU(flux_2d, name="SPEC"))
            result_hdu.append(fits.ImageHDU(errors_2d, name="ERR"))
            result_hdu.append(fits.ImageHDU(wavelengths, name="WAVE"))
            result_hdu.writeto(fits_out, overwrite=True)

            out_files.append(fits_out)

        print(f"{print_msg} [DONE]\n")
        print("Output files:")

        for item in out_files:
            self._update_files(f"CUSTOM_EXTRACT_2D_{nod_ab}", str(item))

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def fit_gaussian(
        self, nod_ab: str = "A", extraction_input: str = "util_extract_2d"
    ) -> None:
        """
        Method for centering the 2D extracted spectra by fitting a
        Gaussian function to the mean of each spectral order. This
        method should be used after extracting the 2D spectra with
        :func:`~pycrires.pipeline.Pipeline.util_extract_2d`.

        Parameters
        ----------
        nod_ab : str
            Nod position of which the extracted spectra will be
            exported to a JSON file ("A" or "B").

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Fit Gaussian")

        # Create output folder

        output_dir = self.product_folder / "fit_gaussian"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List with FITS files that will be processed

        input_folder = self.product_folder / extraction_input
        fits_files = pathlib.Path(input_folder).glob(
            f"cr2res_combined{nod_ab}_*_extr2d.fits"
        )
        n_exp = len(
            list(
                pathlib.Path(input_folder).glob(
                    f"cr2res_combined{nod_ab}_*_extr2d.fits"
                )
            )
        )

        print_msg = ""
        out_files = []

        for fits_idx, fits_item in enumerate(fits_files):
            if fits_idx == 0:
                print(f"Processing exposure #{fits_idx+1}/{n_exp} of nod {nod_ab}:")
            else:
                print(f"\nProcessing exposure #{fits_idx+1}/{n_exp} of nod {nod_ab}:")

            file_name = str(fits_item).split("/")[-2:]
            print(f"   - Input spectrum: product/{file_name[-2]}/{file_name[-1]}")

            with fits.open(fits_item) as hdu_list:
                spec = np.array(hdu_list[1].data)
                err = np.array(hdu_list[2].data)
                wave = np.array(hdu_list[3].data)

            spec_shift = np.zeros(spec.shape)

            def gaussian(x, amp, mean, sigma):
                return amp * np.exp(-0.5 * (x - mean) ** 2 / sigma**2)

            gauss_amp = np.zeros((spec.shape[:2]))
            gauss_mean = np.zeros((spec.shape[:2]))
            gauss_sigma = np.zeros((spec.shape[:2]))

            for det_idx in range(spec.shape[0]):
                for order_idx in range(spec.shape[1]):
                    spec_select = spec[det_idx, order_idx, :, :]
                    y_data = np.median(spec_select, axis=1)

                    if y_data.shape[0] % 2 == 0:
                        x_data = np.linspace(
                            -y_data.shape[0] // 2 + 0.5,
                            y_data.shape[0] // 2 - 0.5,
                            y_data.shape[0],
                        )

                    else:
                        x_data = np.linspace(
                            -(y_data.shape[0] - 1) // 2,
                            (y_data.shape[0] - 1) // 2,
                            y_data.shape[0],
                        )

                    # if np.count_nonzero(y_data) == 0:
                    #     spec_shift[det_idx, order_idx, :, :] = \
                    #         np.full(y_data.shape[0], np.nan)
                    #
                    #     continue

                    try:
                        guess = (np.amax(y_data), 0.0, 1.0)
                        nans = np.isnan(y_data)
                        y_data[nans] = 0
                        result, _ = optimize.curve_fit(
                            gaussian, x_data, y_data, p0=guess
                        )

                        print("\r" + len(print_msg) * " ", end="")

                        print_msg = (
                            f"\r   - Gaussian parameters: "
                            f"amp = {result[0]:.1f}, "
                            f"mean = {result[1]:.1f}, "
                            f"sigma = {result[2]:.1f}"
                        )

                        print(print_msg, end="")

                        spec_shift[det_idx, order_idx, :, :] = ndimage.shift(
                            spec_select, (-result[1], 0.0), order=3, mode="constant"
                        )

                        gauss_amp[det_idx, order_idx] = result[0]
                        gauss_mean[det_idx, order_idx] = result[1]
                        gauss_sigma[det_idx, order_idx] = result[2]

                    except RuntimeError:
                        spec_shift[det_idx, order_idx, :, :] = np.full(
                            spec_select.shape, np.nan
                        )

            print("\r" + len(print_msg) * " ", end="")

            print_msg = (
                f"\r   - Best-fit parameters: "
                f"amp = {np.median(gauss_amp):.1f} +/- {np.std(gauss_amp):.1f}, "
                f"mean = {np.median(gauss_mean):.1f} +/- {np.std(gauss_mean):.1f}, "
                f"sigma = {np.median(gauss_sigma):.1f} +/- {np.std(gauss_sigma):.1f}"
            )

            print(print_msg, end="")

            # for det_idx in range(spec.shape[0]):
            #     for order_idx in range(spec.shape[1]):
            #         g_mean = gauss_mean[det_idx, order_idx]
            #         g_select = gauss_mean[det_idx, order_idx, ]
            #
            #         if g_mean < np.median(g_select) - 3.*np.std(g_select):
            #             spec_shift[det_idx, order_idx, :, wavel_idx] = \
            #                 np.full(y_data.shape[0], np.nan)
            #
            #         elif g_mean > np.median(g_select) + 3.*np.std(g_select):
            #             spec_shift[det_idx, order_idx, :, wavel_idx] = \
            #                 np.full(y_data.shape[0], np.nan)

            hdu_list = fits.HDUList(fits.PrimaryHDU())
            hdu_list.append(fits.ImageHDU(spec_shift, name="SPEC"))
            hdu_list.append(fits.ImageHDU(err, name="ERR"))
            hdu_list.append(fits.ImageHDU(wave, name="WAVE"))

            fits_file = (
                f"{self.path}/product/fit_gaussian/spectra_"
                + f"nod_{nod_ab}_{fits_idx:03d}_center.fits"
            )

            file_name = fits_file.split("/")[-2:]
            print(f"\n   - Output spectrum: product/{file_name[-2]}/{file_name[-1]}")

            hdu_list.writeto(fits_file, overwrite=True)
            out_files.append(fits_file)

        print("\nOutput files:")

        for item in out_files:
            self._update_files(f"FIT_GAUSSIAN_2D_{nod_ab}", str(item))

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def plot_spectra(
        self,
        nod_ab: str = "A",
        telluric: bool = True,
        corrected: bool = False,
        file_id: int = 0,
    ) -> None:
        """
        Method for plotting the extracted spectra.

        Parameters
        ----------
        nod_ab : str
            Nod position of which the extracted spectra are plotted
            ("A" or "B").
        telluric : bool
            Plot a telluric transmission spectrum for comparison. It
            should have been calculated with
            :func:`~pycrires.pipeline.Pipeline.run_skycalc`.
        corrected : bool
            Plot the wavelength-corrected spectra. The output from
            :func:`~pycrires.pipeline.Pipeline.correct_wavelengths`.
        file_id : int
            File ID number from the FITS filename as produced by
            :func:`~pycrires.pipeline.Pipeline.obs_nodding`. The
            numbers consist of three values, starting at 000. To
            select the first file (that contains 000), set
            ``file_id=0``. For the second file, which has 001 in
            its filename, set ``file_id=1``, etc.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Plot spectra")

        if corrected:
            fits_file = (
                f"{self.path}/product/correct_wavelengths/"
                + f"cr2res_obs_nodding_extracted{nod_ab}_"
                + f"{file_id:03d}_corr.fits"
            )

            print(
                f"Spectrum file: cr2res_obs_nodding_extracted{nod_ab}_{file_id:03d}_corr.fits"
            )

        else:
            fits_file = (
                f"{self.path}/product/obs_nodding/cr2res_obs_nodding_"
                + f"extracted{nod_ab}_{file_id:03d}.fits"
            )

            print(
                f"Spectrum file: cr2res_obs_nodding_extracted{nod_ab}_{file_id:03d}.fits"
            )

        print(f"Reading FITS data of nod {nod_ab}...", end="", flush=True)

        spec_data = []

        with fits.open(fits_file) as hdu_list:
            # Loop over 3 detectors
            for i in range(3):
                spec_data.append(hdu_list[i + 1].data)

        print(" [DONE]")

        if telluric:
            tel_spec = self.calib_folder / "run_skycalc/transm_spec.dat"

            if not os.path.exists(tel_spec):
                raise RuntimeError(
                    "Could not find the telluric transmission "
                    "spectrum. Please first run the "
                    "run_skycalc method."
                )

            tel_wavel, tel_transm = np.loadtxt(tel_spec, unpack=True)

        fits_file = (
            self.calib_folder / "molecfit_correct/"
            f"SPECTRUM_TELLURIC_CORR_SCIENCE_{nod_ab}.fits"
        )

        spec_corr = []

        if os.path.exists(fits_file):
            with fits.open(fits_file) as hdu_list:
                hdu_info = hdu_list.info(output=False)
                num_ext = len(hdu_info) - 1

                for i in range(num_ext):
                    spec_corr.append(hdu_list[i + 1].data)

        print("Plotting spectra:")

        count = 0

        for i, det_item in enumerate(spec_data):
            n_spec = len(det_item.columns) // 3

            spec_orders = np.sort([j[:5] for j in det_item.dtype.names if "WL" in j])

            plt.figure(figsize=(8, n_spec * 2))

            for k, spec_item in enumerate(spec_orders):
                ax = plt.subplot(n_spec, 1, n_spec - k)
                ax.minorticks_on()

                if telluric:
                    ax2 = ax.twinx()
                    ax2.minorticks_on()

                wavel = det_item[f"{spec_item}_WL"]
                flux = det_item[f"{spec_item}_SPEC"]
                error = det_item[f"{spec_item}_ERR"]

                flux = np.nan_to_num(flux)
                error = np.nan_to_num(error)

                lower = flux - error
                upper = flux + error

                # indices = np.where((flux != 0.0) & (flux != np.nan))[0]

                ax.plot(wavel, flux, "-", lw=0.5, color="tab:blue")
                ax.fill_between(
                    wavel, lower, upper, color="tab:blue", alpha=0.5, lw=0.0
                )
                if len(spec_corr) > 0:
                    ax.plot(wavel, spec_corr[count], "-", lw=0.3, color="black")
                ax.set_xlabel("Wavelength (nm)", fontsize=13)
                ax.set_ylabel("Flux", fontsize=13)

                xlim = ax.get_xlim()

                if telluric:
                    ax2.plot(tel_wavel, tel_transm, "-", lw=0.5, color="tab:orange")
                    ax2.set_ylabel(r"T$_\lambda$", fontsize=13)
                    ax2.set_xlim(xlim[0], xlim[1])
                    ax2.set_ylim(-0.05, 1.05)

                count += 1

            plt.tight_layout()

            if corrected:
                plot_file = (
                    f"{self.path}/product/correct_wavelengths/"
                    + f"spectra_nod_{nod_ab}_det_{i+1}_corr_"
                    + f"{file_id:03d}.png"
                )
            else:
                plot_file = (
                    f"{self.path}/product/obs_nodding/spectra_nod_"
                    + f"{nod_ab}_det_{i+1}_{file_id:03d}.png"
                )

            file_name = plot_file.split("/")[-2:]
            print(f"   - product/{file_name[-2]}/{file_name[-1]}")

            plt.savefig(plot_file, dpi=300)
            plt.clf()
            plt.close()

    @typechecked
    def export_spectra(
        self,
        nod_ab: str = "A",
        corrected: bool = False,
    ) -> None:
        """
        Method for exporting the 1D extracted spectra to a JSON
        file. After exporting, the data can be read with Python
        from the JSON file into a dictionary. For example,
        reading the spectra of the first exposure:

        >>> import json
        >>> with open('product/obs_nodding/spectra_nod_A_000.json') as json_file:
        ...    data = json.load(json_file)
        >>> print(data.keys())

        Parameters
        ----------
        nod_ab : str
            Nod position of which the extracted spectra will be
            exported to a JSON file ("A" or "B").
        corrected : bool
            Export the wavelength-corrected spectra, so the output from
            :func:`~pycrires.pipeline.Pipeline.correct_wavelengths`.
            If set to ``False``, the extracted spectra from
            :func:`~pycrires.pipeline.Pipeline.obs_nodding` are
            exported.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Export spectra")

        count = 0

        while True:
            if corrected:
                fits_file = (
                    f"{self.path}/product/correct_wavelen"
                    + "gths/cr2res_obs_nodding_extracted"
                    + f"{nod_ab}_{count:03d}_corr.fits"
                )

            else:
                fits_file = (
                    f"{self.path}/product/obs_nodding/"
                    + "cr2res_obs_nodding_extracted"
                    + f"{nod_ab}_{count:03d}.fits"
                )

            if not pathlib.Path(fits_file).exists():
                break

            file_name = fits_file.split("/")[-2:]
            print(f"Reading FITS data: product/{file_name[-2]}/{file_name[-1]}")

            spec_data = []

            with fits.open(fits_file) as hdu_list:
                # Loop over 3 detectors
                for i in range(3):
                    spec_data.append(hdu_list[i + 1].data)

            spec_dict = {}

            for i, det_item in enumerate(spec_data):
                spec_orders = np.sort(
                    [j[:5] for j in det_item.dtype.names if "WL" in j]
                )

                for spec_item in spec_orders:
                    wavel = det_item[f"{spec_item}_WL"]
                    flux = det_item[f"{spec_item}_SPEC"]
                    error = det_item[f"{spec_item}_ERR"]

                    flux = np.nan_to_num(flux)
                    error = np.nan_to_num(error)

                    # indices = np.where((flux != 0.0) & (flux != np.nan) & (error != np.nan))[0]

                    spec_dict[f"det_{i+1}_{spec_item}_WL"] = list(wavel)
                    spec_dict[f"det_{i+1}_{spec_item}_SPEC"] = list(flux)
                    spec_dict[f"det_{i+1}_{spec_item}_ERR"] = list(error)

            if corrected:
                json_out = (
                    self.product_folder
                    / f"correct_wavelengths/spectra_nod_{nod_ab}_{count:03d}.json"
                )
            else:
                json_out = (
                    self.product_folder
                    / f"obs_nodding/spectra_nod_{nod_ab}_{count:03d}.json"
                )

            file_name = str(json_out).split("/")[-2:]
            print(f"Exporting spectra: product/{file_name[-2]}/{file_name[-1]}")

            with open(json_out, "w", encoding="utf-8") as json_file:
                json.dump(spec_dict, json_file, indent=4)

            count += 1

    @typechecked
    def clean_folder(self, keep_product: bool = True) -> None:
        """
        Method for removing all the output that is produced by the
        ``Pipeline`` (so not the raw data).

        Parameters
        ----------
        keep_product : bool
            Keep the `product` folder (``True``) or remove that folder
            as well (``False``).

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Clean pipeline folder")

        files = [self.header_file, self.excel_file, self.json_file]
        folders = [self.config_folder, self.calib_folder]

        if not keep_product:
            folders.append(self.product_folder)

        print("Removing files:")

        for item in files:
            print(f"   - {item}", end="")

            if os.path.exists(item):
                os.remove(item)
                print(" [DONE]")
            else:
                print(" [NOT FOUND]")

        print("\nRemoving folders:")

        for item in folders:
            print(f"   - {item}", end="")

            if os.path.exists(item):
                shutil.rmtree(item)
                print(" [DONE]")
            else:
                print(" [NOT FOUND]")

    @typechecked
    def run_recipes(self) -> None:
        """
        Method for running the full chain of recipes.

        Returns
        -------
        NoneType
            None
        """

        self.rename_files()
        self.extract_header()
        self.cal_dark(verbose=False)
        self.util_calib(calib_type="flat", verbose=False)
        self.util_trace(plot_trace=False, verbose=False)
        self.util_slit_curv(plot_trace=True, verbose=False)
        self.util_extract(calib_type="flat", verbose=False)
        self.util_normflat(verbose=False)
        self.util_calib(calib_type="une", verbose=False)
        self.util_extract(calib_type="une", verbose=False)
        self.util_genlines(verbose=False)
        self.util_wave(calib_type="une", verbose=False)
        self.util_calib(calib_type="fpet", verbose=False)
        self.util_extract(calib_type="fpet", verbose=False)
        self.util_wave(calib_type="fpet", verbose=False)
        self.obs_nodding(verbose=False, correct_bad_pixels=True)
        self.plot_spectra(nod_ab="A", telluric=True, corrected=False, file_id=0)
        self.export_spectra(nod_ab="A", corrected=False)
        self.run_skycalc(pwv=1.0)
        self.correct_wavelengths(nod_ab="A", create_plots=True)
        self.plot_spectra(nod_ab="A", telluric=True, corrected=True, file_id=0)
        self.export_spectra(nod_ab="A", corrected=True)
        self.util_extract_2d(nod_ab="A", verbose=False, use_corr_wavel=True)
        # self.molecfit_input(nod_ab="A")
        # self.molecfit_model(nod_ab="A", verbose=False)
        # self.molecfit_calctrans(nod_ab="A", verbose=False)
        # self.molecfit_correct(nod_ab="A", verbose=False)
