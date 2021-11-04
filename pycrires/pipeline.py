"""
Module with the pipeline functionalities of ``pycrires``.
"""

import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
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
from scipy.ndimage.filters import gaussian_filter
from typeguard import typechecked


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
            # Print the available EsoRex recipes from CRIRES+

            esorex = ["esorex", "--recipes"]

            with subprocess.Popen(
                esorex, cwd=self.config_folder, stdout=subprocess.PIPE, encoding="utf-8"
            ) as proc:
                output, _ = proc.communicate()

            print("Available EsoRex recipes for CRIRES+:")

            for item in output.split("\n"):
                if item.replace(" ", "")[:7] == "cr2res_":
                    print(f"   -{item}")

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
    def _print_info(self) -> None:
        """
        Internal method for printing some details about the
        observations.

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
    def _update_files(self, sof_key: str, file_name: str) -> None:
        """
        Internal method for updating the dictionary with file
        names and related Set Of Files (SOF) keywords.

        Parameters
        ----------
        sof_key : str
            Set Of Files (SOF) keyword of ``file_name``.
        file_name : str
            Absolute path of the file.

        Returns
        -------
        NoneType
            None
        """

        # Print filename and SOF keyword

        file_split = file_name.split("/")

        if file_split[-2] == "raw":
            file_print = file_split[-2] + "/" + file_split[-1]
            print(f"   - {file_print} {sof_key}")

        elif file_split[-3] == "calib":
            file_print = file_split[-3] + "/" + file_split[-2] + "/" + file_split[-1]
            print(f"   - {file_print} {sof_key}")

        elif file_split[-2] == "product":
            file_print = file_split[-2] + "/" + file_split[-1]
            print(f"   - {file_print} {sof_key}")

        else:
            print(f"   - {file_name} {sof_key}")

        # Get FITS header

        if file_name.endswith(".fits"):
            header = fits.getheader(file_name)
        else:
            header = None

        file_dict = {}

        if header is not None and "ESO DET SEQ1 DIT" in header:
            file_dict["DIT"] = header["ESO DET SEQ1 DIT"]
        else:
            file_dict["DIT"] = None

        if header is not None and "ESO INS WLEN ID" in header:
            file_dict["WLEN"] = header["ESO INS WLEN ID"]
        else:
            file_dict["WLEN"] = None

        if sof_key in self.file_dict:
            if file_name not in self.file_dict[sof_key]:
                self.file_dict[sof_key][file_name] = file_dict
        else:
            self.file_dict[sof_key] = {file_name: file_dict}

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

            if eso_recipe == "molecfit_model":
                with open(config_file, "r", encoding="utf-8") as open_config:
                    rc_lines = open_config.read().splitlines()

                rc_lines[32] = "USE_INPUT_KERNEL=FALSE"
                rc_lines[50] = "LIST_MOLEC=H2O,CO2,O3,CO"
                rc_lines[63] = "FIT_MOLEC=1,1,1,1"
                rc_lines[78] = "REL_COL=1.0,1.0,1.0,1.0"
                rc_lines[93] = "WAVE_INCLUDE=0.5,5.0"
                rc_lines[147] = "COLUMN_LAMBDA=WAVE"
                rc_lines[151] = "COLUMN_FLUX=SPEC"
                rc_lines[155] = "COLUMN_DFLUX=ERR"
                rc_lines[169] = "WLG_TO_MICRON=0.001"

                with open(config_file, "w", encoding="utf-8") as open_config:
                    open_config.write("\n".join(rc_lines))

            if not verbose:
                print(" [DONE]")

    @typechecked
    def _download_archive(self, dpr_type: str, det_dit: Optional[float]):
        """
        Internal method for downloading data from the ESO Science Archive.

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
        Internal method for plotting the raw data of a specified ``DPR.CATG``
        together with the traces of the spectral orders. The content of this
        method has been adapted from the ``cr2res_show_trace_curv.py`` code
        that is included with the EsoRex pipeline for CRIRES+ (see
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
                                x[j] = i + y_item * b + y_item ** 2 * c

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
        self._print_info()

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
            of the telluric transmission lines which can be seen
            when plotting the spectra with
            :meth:`~pycrires.pipeline.Pipeline.plot_spectra` and
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

        if self.header_data["INS.WLEN.ID"][0][0] == "Y":
            sky_calc["wmin"] = 500.0  # (nm)
            sky_calc["wmax"] = 1500.0  # (nm)

        elif self.header_data["INS.WLEN.ID"][0][0] == "J":
            sky_calc["wmin"] = 800.0  # (nm)
            sky_calc["wmax"] = 2000.0  # (nm)

        elif self.header_data["INS.WLEN.ID"][0][0] == "H":
            sky_calc["wmin"] = 1000.0  # (nm)
            sky_calc["wmax"] = 2500.0  # (nm)

        elif self.header_data["INS.WLEN.ID"][0][0] == "K":
            sky_calc["wmin"] = 1500.0  # (nm)
            sky_calc["wmax"] = 3000.0  # (nm)

        else:
            wlen_id = self.header_data["INS.WLEN.ID"][science_index][0]

            raise NotImplementedError(
                f"The wavelength range for {wlen_id} is not yet implemented."
            )

        sky_calc["wgrid_mode"] = "fixed_spectral_resolution"
        sky_calc["wres"] = 2e5
        sky_calc["pwv"] = pwv

        print(f"  - Wavelength range (nm) = {sky_calc['wmin']} - {sky_calc['wmax']}")
        print(f"  - Sampling resolution = {sky_calc['wres']}")
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

        sky_spec["flux"] = gaussian_filter(
            sky_spec["flux"], sigma=sigma_lsf_gauss_filter, mode="nearest"
        )

        sky_spec["trans"] = gaussian_filter(
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

        self._print_section("Create master dark", recipe_name="cr2res_cal_dark")

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

        self._print_section("Create master flat", recipe_name="cr2res_cal_flat")

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
                            file_name = key.split("/")[-1]
                            print(f"   - calib/{file_name} CAL_DARK_MASTER")
                            sof_open.write(f"{key} CAL_DARK_MASTER\n")
                            file_found = True

                if not file_found:
                    self._download_archive("DARK", dit_item)

                # Find bad pixel map

                file_found = False

                if "CAL_DARK_BPM" in self.file_dict:
                    for key, value in self.file_dict["CAL_DARK_BPM"].items():
                        if not file_found and value["DIT"] == dit_item:
                            file_name = key.split("/")[-1]
                            print(f"   - calib/{file_name} CAL_DARK_BPM")
                            sof_open.write(f"{key} CAL_DARK_BPM\n")
                            file_found = True

                if not file_found:
                    warnings.warn(
                        f"There is not a bad pixel map with DIT = {dit_item} s."
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
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name} UTIL_TRACE_TW")
                        sof_open.write(f"{key} UTIL_TRACE_TW\n")
                        file_found = True

            if "CAL_WAVE_TW" in self.file_dict:
                for key in self.file_dict["CAL_WAVE_TW"]:
                    if not file_found:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name} CAL_WAVE_TW")
                        sof_open.write(f"{key} CAL_WAVE_TW\n")
                        file_found = True

            if "CAL_FLAT_TW" in self.file_dict:
                for key in self.file_dict["CAL_FLAT_TW"]:
                    if not file_found:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name} CAL_FLAT_TW")
                        sof_open.write(f"{key} CAL_FLAT_TW\n")
                        file_found = True

            if not file_found:
                raise RuntimeError("Could not find a TraceWave file.")

            # Find emission lines file

            file_found = False

            if "EMISSION_LINES" in self.file_dict:
                for key in self.file_dict["EMISSION_LINES"]:
                    if not file_found:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name} EMISSION_LINES")
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
            self._print_section("Create master flat", recipe_name="cr2res_util_calib")

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
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name} CAL_DARK_MASTER")
                        sof_open.write(f"{key} CAL_DARK_MASTER\n")
                        file_found = True

            if not file_found:
                self._download_archive("DARK", dit_item)

            # Find bad pixel map

            file_found = False

            if "CAL_DARK_BPM" in self.file_dict:
                for key, value in self.file_dict["CAL_DARK_BPM"].items():
                    if not file_found and value["DIT"] == dit_item:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name} CAL_DARK_BPM")
                        sof_open.write(f"{key} CAL_DARK_BPM\n")
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
                            file_name = key.split("/")[-1]
                            print(
                                f"   - calib/util_calib_flat/{file_name} UTIL_MASTER_FLAT"
                            )
                            sof_open.write(f"{key} UTIL_MASTER_FLAT\n")
                            file_found = True

                if "CAL_FLAT_MASTER" in self.file_dict:
                    for key, value in self.file_dict["CAL_FLAT_MASTER"].items():
                        if not file_found:
                            file_name = key.split("/")[-1]
                            print(f"   - calib/cal_flat/{file_name} CAL_FLAT_MASTER")
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
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name} CAL_DETLIN_COEFFS")
                        sof_open.write(f"{key} CAL_DETLIN_COEFFS\n")
                        file_found = True

            if not file_found:
                warnings.warn("Could not find CAL_DETLIN_COEFFS.")

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_util_calib", "util_calib", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "util_calib.rc"

        if calib_type in ["flat", "fpet"]:
            esorex = [
                "esorex",
                f"--recipe-config={config_file}",
                f"--output-dir={output_dir}",
                "cr2res_util_calib",
                "--collapse=MEAN",
                sof_file,
            ]

        elif calib_type == "une":
            esorex = [
                "esorex",
                f"--recipe-config={config_file}",
                f"--output-dir={output_dir}",
                "cr2res_util_calib",
                "--collapse=MEAN",
                "--subtract_nolight_rows=TRUE",
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
                        file_name = key.split("/")[-1]
                        print(f"   - calib/util_calib_flat/{file_name} UTIL_CALIB")
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
                        file_name = key.split("/")[-1]
                        print(f"   - calib/util_trace/{file_name} UTIL_TRACE_TW")
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
                "Extract blaze spectrum", recipe_name="cr2res_util_extract"
            )

        elif calib_type == "une":
            self._print_section(
                "Extract UNE spectrum", recipe_name="cr2res_util_extract"
            )

        elif calib_type == "fpet":
            self._print_section("Extract FPET frame", recipe_name="cr2res_util_extract")

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
                            file_name = key.split("/")[-1]
                            print(
                                f"   - calib/util_calib_{calib_type}/{file_name} UTIL_CALIB"
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
        #                 file_name = key.split("/")[-1]
        #                 print(f"   - calib/util_trace/{file_name} UTIL_TRACE_TW")
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
                        file_name = key.split("/")[-1]
                        print(
                            f"   - calib/util_wave_une/{file_name} UTIL_SLIT_CURV_TW"
                        )
                        sof_open.write(f"{key} UTIL_SLIT_CURV_TW\n")
                        file_found = True

        if "UTIL_SLIT_CURV_TW" in self.file_dict:
            for key in self.file_dict["UTIL_SLIT_CURV_TW"]:
                if not file_found:
                    with open(sof_file, "a", encoding="utf-8") as sof_open:
                        file_name = key.split("/")[-1]
                        print(
                            f"   - calib/util_slit_curv/{file_name} UTIL_SLIT_CURV_TW"
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

        self._create_config("cr2res_util_extract", "util_extract", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "util_extract.rc"

        if calib_type == "flat":
            esorex = [
                "esorex",
                f"--recipe-config={config_file}",
                f"--output-dir={output_dir}",
                "cr2res_util_extract",
                "--smooth_slit=3",
                "-smooth_spec=2.0e-7",
                sof_file,
            ]

        else:
            esorex = [
                "esorex",
                f"--recipe-config={config_file}",
                f"--output-dir={output_dir}",
                "cr2res_util_extract",
                "--smooth_slit=3",
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
                        file_name = key.split("/")[-1]
                        print(f"   - calib/util_calib_flat/{file_name} UTIL_CALIB")
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
                        file_name = key.split("/")[-1]
                        print(
                            f"   - calib/util_extract_flat/{file_name} UTIL_SLIT_MODEL"
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

        code_dir = os.path.dirname(os.path.realpath(__file__))

        wavel_set = self.header_data["INS.WLEN.ID"][0]

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
    def util_wave(self, calib_type: str, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_util_wave``.

        Parameters
        ----------
        calib_type : str
            Calibration type ("une" or "fpet").
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        # Check if the wavelength solution was already determined with UNE

        check_une = False

        if "UTIL_WAVE_TW" in self.file_dict:
            for key in self.file_dict["UTIL_WAVE_TW"]:
                if key.split("/")[-2] == "util_wave_une":
                    check_une = True

        if calib_type == "une" or not check_une:
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
                            file_name = key.split("/")[-1]
                            print(f"   - calib/util_genlines/{file_name} EMISSION_LINES")
                            sof_open.write(f"{key} EMISSION_LINES\n")
                            file_found = True

            else:
                raise RuntimeError(
                    "The EMISSION_LINES file is not found in "
                    "the 'calib/genlines' folder. Please first "
                    "run the util_genlines method."
                )

        # Find UTIL_CALIB file

        file_found = False

        if "UTIL_EXTRACT_1D" in self.file_dict:
            for key in self.file_dict["UTIL_EXTRACT_1D"]:
                if (
                    not file_found
                    and key.split("/")[-2] == f"util_extract_{calib_type}"
                ):
                    with open(sof_file, "a", encoding="utf-8") as sof_open:
                        file_name = key.split("/")[-1]
                        print(
                            f"   - calib/util_extract_{calib_type}/"
                            f"{file_name} UTIL_EXTRACT_1D"
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

        if calib_type == "une" or not check_une:
            file_found = False

            if "UTIL_SLIT_CURV_TW" in self.file_dict:
                for key in self.file_dict["UTIL_SLIT_CURV_TW"]:
                    if not file_found:
                        with open(sof_file, "a", encoding="utf-8") as sof_open:
                            file_name = key.split("/")[-1]
                            print(
                                f"   - calib/util_slit_curv/{file_name} UTIL_SLIT_CURV_TW"
                            )
                            sof_open.write(f"{key} UTIL_SLIT_CURV_TW\n")
                            file_found = True

            else:
                raise RuntimeError(
                    "The UTIL_SLIT_CURV_TW file is not found "
                    "in the 'calib/util_slit_curv' folder. "
                    "Please first run the util_slit_curv method."
                )

        # Find UTIL_WAVE_TW file

        if calib_type == "fpet" and check_une:
            file_found = False

            if "UTIL_WAVE_TW" in self.file_dict:
                for key in self.file_dict["UTIL_WAVE_TW"]:
                    if not file_found and key.split("/")[-2] == "util_wave_une":
                        with open(sof_file, "a", encoding="utf-8") as sof_open:
                            file_name = key.split("/")[-1]
                            print(f"   - calib/util_wave_une/{file_name} UTIL_WAVE_TW")
                            sof_open.write(f"{key} UTIL_WAVE_TW\n")
                            file_found = True

            else:
                raise RuntimeError(
                    "The UTIL_WAVE_TW file is not found in the "
                    "'calib/util_wave_une' folder. Please first "
                    "run the util_wave method with calib_type='une'."
                )

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_util_wave", "util_wave", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "util_wave.rc"

        if calib_type == "une":
            esorex = [
                "esorex",
                f"--recipe-config={config_file}",
                f"--output-dir={output_dir}",
                "cr2res_util_wave",
                "--wl_method=XCORR",
                "--wl_degree=0",
                "--keep",
                "--wl_err=0.1",
                "--fallback",
                sof_file,
            ]

        elif calib_type == "fpet" and check_une:
            esorex = [
                "esorex",
                f"--recipe-config={config_file}",
                f"--output-dir={output_dir}",
                "cr2res_util_wave",
                "--wl_method=ETALON",
                "--wl_degree=4",
                "--fallback",
                sof_file,
            ]

        elif calib_type == "fpet" and not check_une:
            esorex = [
                "esorex",
                f"--recipe-config={config_file}",
                f"--output-dir={output_dir}",
                "cr2res_util_wave",
                "--wl_method=XCORR",
                "--wl_degree=0",
                "--keep",
                "--wl_err=0.1",
                "--fallback",
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
    def obs_nodding(self, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_obs_nodding``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Process nodding frames", recipe_name="cr2res_obs_nodding")

        indices = self.header_data["DPR.CATG"] == "SCIENCE"

        # for i, item in enumerate(self.header_data["DPR.TYPE"]):
        #     if item != "OBJECT":
        #         indices[i] = False

        # Check unique DIT

        unique_dit = set()
        for item in self.header_data[indices]["DET.SEQ1.DIT"]:
            unique_dit.add(item)

        print(f"Unique DIT values: {unique_dit}\n")

        # Create SOF file

        print("Creating SOF file:")

        sof_file = pathlib.Path(self.product_folder / "files.sof")

        with open(sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                file_path = f"{self.path}/raw/{item}"
                header = fits.getheader(file_path)

                if "ESO DPR TECH" in header:
                    if header["ESO DPR TECH"] == "SPECTRUM,NODDING,OTHER":
                        sof_open.write(f"{file_path} OBS_NODDING_OTHER\n")
                        self._update_files("OBS_NODDING_OTHER", file_path)

                    elif header["ESO DPR TECH"] == "SPECTRUM,NODDING,JITTER":
                        sof_open.write(f"{file_path} OBS_NODDING_JITTER\n")
                        self._update_files("OBS_NODDING_JITTER", file_path)

                else:
                    raise RuntimeError(
                        f"Could not find ESO.DPR.TECH in " f"the header of {item}."
                    )

            # Find UTIL_MASTER_FLAT or CAL_FLAT_MASTER file

            file_found = False

            if "UTIL_MASTER_FLAT" in self.file_dict:
                for key in self.file_dict["UTIL_MASTER_FLAT"]:
                    if not file_found:
                        file_name = key.split("/")[-1]
                        print(
                            f"   - calib/util_calib_flat/{file_name} UTIL_MASTER_FLAT"
                        )
                        sof_open.write(f"{key} UTIL_MASTER_FLAT\n")
                        file_found = True

            if "CAL_FLAT_MASTER" in self.file_dict:
                for key in self.file_dict["CAL_FLAT_MASTER"]:
                    if not file_found:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/cal_flat/{file_name} CAL_FLAT_MASTER")
                        sof_open.write(f"{key} CAL_FLAT_MASTER\n")
                        file_found = True

            if not file_found:
                warnings.warn("Could not find a master flat.")

            # Find CAL_DARK_BPM file

            file_found = False

            if "CAL_DARK_BPM" in self.file_dict:
                for key in self.file_dict["CAL_DARK_BPM"]:
                    if not file_found:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/cal_dark/{file_name} CAL_DARK_BPM")
                        sof_open.write(f"{key} CAL_DARK_BPM\n")
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
                            file_name = key.split("/")[-1]
                            print(
                                f"   - calib/util_wave_{calib_type}/{file_name} UTIL_WAVE_TW"
                            )
                            sof_open.write(f"{key} UTIL_WAVE_TW\n")
                            file_found = True

            # if "CAL_WAVE_TW" in self.file_dict:
            #     for key in self.file_dict["CAL_WAVE_TW"]:
            #         if not file_found:
            #             file_name = key.split("/")[-1]
            #             print(f"   - calib/{file_name} CAL_WAVE_TW")
            #             sof_open.write(f"{key} CAL_WAVE_TW\n")
            #             file_found = True
            #
            # if "CAL_FLAT_TW" in self.file_dict:
            #     for key in self.file_dict["CAL_FLAT_TW"]:
            #         if not file_found:
            #             file_name = key.split("/")[-1]
            #             print(f"   - calib/{file_name} CAL_FLAT_TW")
            #             sof_open.write(f"{key} CAL_FLAT_TW\n")
            #             file_found = True

            if not file_found:
                warnings.warn("Could not find file with TraceWave table.")

            # Find CAL_DETLIN_COEFFS file

            file_found = False

            if "CAL_DETLIN_COEFFS" in self.file_dict:
                for key in self.file_dict["CAL_DETLIN_COEFFS"]:
                    if not file_found:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name} CAL_DETLIN_COEFFS")
                        sof_open.write(f"{key} CAL_DETLIN_COEFFS\n")
                        file_found = True

            if not file_found:
                warnings.warn("Could not find CAL_DETLIN_COEFFS.")

        # Create EsoRex configuration file if not found

        self._create_config("cr2res_obs_nodding", "obs_nodding", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "obs_nodding.rc"

        esorex = [
            "esorex",
            f"--recipe-config={config_file}",
            f"--output-dir={self.product_folder}",
            "cr2res_obs_nodding",
            sof_file,
        ]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=self.product_folder, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]\n")

        # Update file dictionary with output files

        print("Output files:")

        fits_file = self.product_folder / "cr2res_obs_nodding_extractedA.fits"
        self._update_files("OBS_NODDING_EXTRACTA", str(fits_file))

        fits_file = self.product_folder / "cr2res_obs_nodding_extractedB.fits"
        self._update_files("OBS_NODDING_EXTRACTB", str(fits_file))

        fits_file = self.product_folder / "cr2res_obs_nodding_combinedA.fits"
        self._update_files("OBS_NODDING_COMBINEDA", str(fits_file))

        fits_file = self.product_folder / "cr2res_obs_nodding_combinedB.fits"
        self._update_files("OBS_NODDING_COMBINEDB", str(fits_file))

        fits_file = self.product_folder / "cr2res_obs_nodding_modelA.fits"
        self._update_files("OBS_NODDING_SLITMODELA", str(fits_file))

        fits_file = self.product_folder / "cr2res_obs_nodding_modelB.fits"
        self._update_files("OBS_NODDING_SLITMODELB", str(fits_file))

        fits_file = self.product_folder / "cr2res_obs_nodding_slitfuncA.fits"
        self._update_files("OBS_NODDING_SLITFUNCA", str(fits_file))

        fits_file = self.product_folder / "cr2res_obs_nodding_slitfuncB.fits"
        self._update_files("OBS_NODDING_SLITFUNCB", str(fits_file))

        # fits_file = self.product_folder / "cr2res_obs_nodding_throughput.fits"
        # self._update_files("OBS_NODDING_THROUGHPUT", str(fits_file))

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def molecfit_input(self, nod_ab: str = "A") -> None:
        """
        Method for converting the output files with the extracted
        spectra into input files for `Molecfit`. The content of this
        method has been adapted from the ``cr2res_drs2molecfit.py``
        code that is included with the EsoRex pipeline for CRIRES+
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

        print(f"Input file: product/cr2res_obs_nodding_extracted{nod_ab}.fits")
        fits_file = f"{self.path}/product/cr2res_obs_nodding_extracted{nod_ab}.fits"

        print("\nOutput files:")

        with fits.open(fits_file) as hdu_list:
            # Use the primary HDU with header in output files
            hdul_output = fits.HDUList([hdu_list[0]])
            hdul_winc = fits.HDUList([hdu_list[0]])
            hdul_atm = fits.HDUList([hdu_list[0]])

            # Initialize empty lists
            wmin, wmax, map2chip, atm_parms_ext = [], [], [], []

            # Initialize counter for order/detector
            count = 1

            # Loop over 3 detectors
            for i_det in range(3):
                # Get detector data
                data = hdu_list[f"CHIP{i_det+1}.INT1"].data

                # Find all spectral orders
                spec_ord = np.sort([i[0:6] for i in data.dtype.names if "WL" in i])

                # loop over orders
                for item in spec_ord:
                    # isolate WAVE, SPEC, and ERR for given order/detector
                    wavel = hdu_list[f"CHIP{i_det+1}.INT1"].data[item + "WL"]
                    spec = hdu_list[f"CHIP{i_det+1}.INT1"].data[item + "SPEC"]
                    err = hdu_list[f"CHIP{i_det+1}.INT1"].data[item + "ERR"]

                    # create fits column for WAVE
                    col1 = fits.Column(name="WAVE", format="D", array=wavel)

                    # create fits column for SPEC
                    col2 = fits.Column(name="SPEC", format="D", array=spec)

                    # create fits column for ERR
                    col3 = fits.Column(name="ERR", format="D", array=err)

                    # create fits table HDU with WAVE SPEC ERR
                    table_hdu = fits.BinTableHDU.from_columns([col1, col2, col3])

                    # append table HDU to output HDU list
                    hdul_output.append(table_hdu)

                    # append wmin for given order to wmin array and convert to micron
                    wmin.append(np.min(wavel) * 0.001)

                    # append wmax for giver order to wmax array and convert to micron
                    wmax.append(np.max(wavel) * 0.001)

                    # append order counter to map2chip
                    map2chip.append(count)

                    if count == 1:
                        atm_parms_ext.append(0)
                    else:
                        atm_parms_ext.append(1)

                    count += 1

            # setup the columns for the WAVE_INCLUDE output file
            wmin_winc = fits.Column(name="LOWER_LIMIT", format="D", array=wmin)
            wmax_winc = fits.Column(name="UPPER_LIMIT", format="D", array=wmax)
            map2chip_winc = fits.Column(
                name="MAPPED_TO_CHIP", format="I", array=map2chip
            )
            # creates WAVE_INCLUDE table HDU and appends to the HDU list
            table_hdu_winc = fits.BinTableHDU.from_columns(
                [wmin_winc, wmax_winc, map2chip_winc]
            )
            hdul_winc.append(table_hdu_winc)

            print(f"   - calib/molefit_files/SCIENCE_{nod_ab}.fits")
            hdul_output.writeto(output_dir / f"SCIENCE_{nod_ab}.fits", overwrite=True)

            print(f"   - calib/molefit_files/WAVE_INCLUDE_{nod_ab}.fits")
            hdul_winc.writeto(
                output_dir / f"WAVE_INCLUDE_{nod_ab}.fits", overwrite=True
            )

            print(f"   - calib/molefit_files/MAPPING_ATMOSPHERIC_{nod_ab}.fits")
            col1_atm = fits.Column(
                name="ATM_PARAMETERS_EXT", format="I", array=atm_parms_ext
            )
            table_hdu_atm = fits.BinTableHDU.from_columns([col1_atm])
            hdul_atm.append(table_hdu_atm)
            hdul_atm.writeto(
                output_dir / f"MAPPING_ATMOSPHERIC_{nod_ab}.fits", overwrite=True
            )

    @typechecked
    def molecfit_model(self, nod_ab: str = "A", verbose: bool = True) -> None:
        """
        Method for running ``molecfit_model``.

        Parameters
        ----------
        nod_ab : str
            Nod position of which the extracted spectra are plotted ("A" or "B").
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Run Molecfit", recipe_name="molecfit_model")

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

        # Create EsoRex configuration file if not found

        self._create_config("molecfit_model", "molecfit_model", verbose)

        # Run EsoRex

        print()

        config_file = self.config_folder / "molecfit_model.rc"

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
    def plot_spectra(self, nod_ab: str = "A", telluric: bool = True) -> None:
        """
        Method for plotting the extracted spectra.

        Parameters
        ----------
        nod_ab : str
            Nod position of which the extracted spectra are plotted ("A" or "B").

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Plot spectra")

        fits_file = f"{self.path}/product/cr2res_obs_nodding_extracted{nod_ab}.fits"

        print(f"Spectrum file: cr2res_obs_nodding_extracted{nod_ab}.fits")

        print(f"Reading FITS of nod {nod_ab}...", end="", flush=True)

        spec_data = []
        spec_header = []

        with fits.open(fits_file) as hdu_list:
            # Skip the empty PrimaryHDU
            for item in hdu_list[1:]:
                spec_data.append(item.data)
                spec_header.append(item.header)

        print(" [DONE]")

        if telluric:
            tel_wavel, tel_transm = np.loadtxt(
                self.calib_folder / "run_skycalc/transm_spec.dat", unpack=True
            )

        print("Plotting spectra...", end="", flush=True)

        for i, det_item in enumerate(spec_data):
            n_spec = len(det_item.columns) // 3

            spec_order = []
            for j in range(1, 20):
                order_int = str(j).zfill(2)

                if f"{order_int}_01_WL" in det_item.columns.names:
                    spec_order.append(f"{order_int}_01")

            if telluric:
                plt.figure(figsize=(8, n_spec * 3))
            else:
                plt.figure(figsize=(8, n_spec * 2))

            for j, spec_item in enumerate(spec_order):
                if telluric:
                    plt.subplot(n_spec * 2, 1, n_spec * 2 - j * 2)
                else:
                    plt.subplot(n_spec, 1, n_spec - j)

                wavel = det_item[f"{spec_item}_WL"]
                flux = det_item[f"{spec_item}_SPEC"]
                error = det_item[f"{spec_item}_ERR"]

                lower = flux - error
                upper = flux + error

                indices = np.where((flux != 0.0) & (flux != np.nan))[0]

                plt.plot(wavel[indices], flux[indices], "-", lw=0.5, color="tab:blue")
                plt.fill_between(
                    wavel[indices],
                    lower[indices],
                    upper[indices],
                    color="tab:blue",
                    alpha=0.5,
                    lw=0.0,
                )
                plt.xlabel("Wavelength (nm)", fontsize=13)
                plt.ylabel("Flux", fontsize=13)
                plt.minorticks_on()
                xlim = plt.xlim()

                if telluric:
                    plt.subplot(n_spec * 2, 1, n_spec * 2 - j * 2 - 1)
                    plt.plot(tel_wavel, tel_transm, "-", lw=0.5, color="tab:orange")
                    plt.xlabel("Wavelength (nm)", fontsize=13)
                    plt.ylabel(r"T$_\lambda$", fontsize=13)
                    plt.xlim(xlim[0], xlim[1])
                    plt.ylim(-0.05, 1.05)
                    plt.minorticks_on()

            plt.tight_layout()
            plt.savefig(
                f"{self.path}/product/spectra_nod_{nod_ab}_det_{i+1}.png", dpi=300
            )
            plt.clf()
            plt.close()

        print(" [DONE]")

    @typechecked
    def clean_folder(self) -> None:
        """
        Method for removing all the output that is produced by the
        ``Pipeline`` (so not the raw data).

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Clean pipeline folder")

        files = [self.header_file, self.excel_file, self.json_file]
        folders = [self.config_folder, self.calib_folder, self.product_folder]

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
        self.obs_nodding(verbose=False)
        self.run_skycalc(pwv=1.0)
        self.plot_spectra(nod_ab="A", telluric=True)
        self.plot_spectra(nod_ab="B", telluric=True)
        self.molecfit_input(nod_ab="A")
