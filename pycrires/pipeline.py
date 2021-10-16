"""
Module with the main pipeline class.
"""

import json
import logging
import os
import pathlib
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import skycalc_ipy

from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib import pyplot as plt
from typeguard import typechecked


log_book = logging.getLogger(__name__)


class Pipeline:
    """
    Class for the data reduction pipeline.
    """

    @typechecked
    def __init__(self, path: str) -> None:
        """
        Parameters
        ----------
        path : str
            Path of the main reduction folder. The main folder should
            contain a subfolder called ``raw`` where raw data from the
            ESO archive are stored.

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

        self.sof_file = pathlib.Path(self.path / "calib/files.sof")
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

        # Create directory for temporary files

        self.tmp_folder = pathlib.Path(self.path / "tmp")

        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)

        # Create directory for calibration files

        self.calib_folder = pathlib.Path(self.path / "calib")

        if not os.path.exists(self.calib_folder):
            os.makedirs(self.calib_folder)

        # Create directory for product files

        self.product_folder = pathlib.Path(self.path / "product")

        if not os.path.exists(self.product_folder):
            os.makedirs(self.product_folder)

        # Test if esorex is installed

        if shutil.which("esorex") is None:
            warnings.warn(
                "Esorex is not accessible from the command line. "
                "Please make sure that the ESO pipeline is correctly "
                "installed and included in the PATH variable."
            )

        else:
            # Print the available esorex recipes from CRIRES+

            esorex = ["esorex", "--recipes"]

            with subprocess.Popen(
                esorex, stdout=subprocess.PIPE, encoding="utf-8"
            ) as proc:
                output, _ = proc.communicate()

            print("Available esorex recipes for CRIRES+:")

            for item in output.split("\n"):
                if item.replace(" ", "")[:7] == "cr2res_":
                    print(f"   -{item}")

    @staticmethod
    @typechecked
    def _print_section(
        sect_title: str, bound_char: str = "-", extra_line: bool = True
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

        if "RA" in self.header_data and "DEC" in self.header_data:
            ra_mean = np.mean(self.header_data["RA"])
            dec_mean = np.mean(self.header_data["DEC"])

            target_coord = SkyCoord(ra_mean, dec_mean, unit="deg", frame="icrs")

            ra_dec = target_coord.to_string("hmsdms")

            print(f"RA Dec = {ra_dec}")

        for key, value in check_key.items():
            header = self.header_data[key].to_numpy()

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
                count_files = np.sum(self.header_data["OBS.ID"] == item)
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
                    f"The ARCFILE keyword is not found in " f"the header of {item.name}"
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
    def update_files(self, sof_key: str, file_name: str) -> None:
        """
        Method for

        Parameters
        ----------
        sof_key : str
            SOF keyword for the file.
        file_name : str
            Absolute path of the file.

        Returns
        -------
        NoneType
            None
        """

        header = fits.getheader(file_name)

        file_dict = {}

        if "ESO DET SEQ1 DIT" in header:
            file_dict["DIT"] = header["ESO DET SEQ1 DIT"]
        else:
            file_dict["DIT"] = None

        if "ESO INS WLEN ID" in header:
            file_dict["WLEN"] = header["ESO INS WLEN ID"]
        else:
            file_dict["WLEN"] = None

        if sof_key in self.file_dict:
            if file_name not in self.file_dict[sof_key]:
                self.file_dict[sof_key][file_name] = file_dict
        else:
            self.file_dict[sof_key] = {file_name: file_dict}

    @typechecked
    def run_skycalc(self) -> None:
        """
        Method for running the Python wrapper of SkyCalc
        (see https://skycalc-ipy.readthedocs.io).

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Run SkyCalc")

        temp_file = self.calib_folder / "skycalc_temp.fits"

        print("Retrieving telluric spectrum with SkyCalc...", end="", flush=True)

        sky_calc = skycalc_ipy.SkyCalc()
        sky_spec = sky_calc.get_sky_spectrum(filename=temp_file)

        print(" [DONE]\n")

        emission_spec = np.column_stack(
            (1e3 * sky_spec["lam"], 1e-3 * sky_spec["flux"])
        )
        header = "Wavelength (nm) - Flux (ph arcsec-2 m-2 s-1 nm-1)"

        print("Storing spectrum: calib/sky_spec.dat")
        np.savetxt(self.calib_folder / "sky_spec.dat", emission_spec, header=header)

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

        self._print_section("Create master dark")

        indices = self.header_data["DPR.TYPE"] == "DARK"

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

        with open(self.sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                print(f"   - raw/{item} DARK")
                sof_open.write(f"{self.path}/raw/{item} DARK\n")
                self.update_files("DARK", f"{self.path}/raw/{item}")

        # Check if any dark frames were found

        if "DARK" not in self.file_dict:
            raise RuntimeError("The \'raw\' folder does not contain "
                               "any DPR.TYPE=DARK files.")

        # Run EsoRex

        print()

        esorex = ["esorex", "cr2res_cal_dark", self.sof_file]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        if not verbose:
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=self.calib_folder, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]")

        # Update file dictionary with master dark

        fits_files = pathlib.Path(self.path / "calib").glob(
            "cr2res_cal_dark_*master.fits"
        )

        for item in fits_files:
            self.update_files("CAL_DARK_MASTER", str(item))

        # Update file dictionary with bad pixel map

        fits_files = pathlib.Path(self.path / "calib").glob("cr2res_cal_dark_*bpm.fits")

        for item in fits_files:
            self.update_files("CAL_DARK_BPM", str(item))

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

        self._print_section("Create master flat and detect traces")

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

            with open(self.sof_file, "w", encoding="utf-8") as sof_open:
                for item in self.header_data[indices]["ORIGFILE"]:
                    index = self.header_data.index[
                        self.header_data["ORIGFILE"] == item
                    ][0]
                    flat_dit = self.header_data.iloc[index]["DET.SEQ1.DIT"]

                    if flat_dit == dit_item:
                        print(f"   - raw/{item} FLAT")
                        file_path = f"{self.path}/raw/{item}"
                        sof_open.write(f"{file_path} FLAT\n")
                        self.update_files("FLAT", file_path)

                # Find master dark

                file_found = False

                for key, value in self.file_dict["CAL_DARK_MASTER"].items():
                    if not file_found and value["DIT"] == dit_item:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name} CAL_DARK_MASTER")
                        sof_open.write(f"{key} CAL_DARK_MASTER\n")
                        file_found = True

                if not file_found:
                    warnings.warn(
                        f"There is not a master dark with DIT = {dit_item} s."
                        f"For best results, please download a DPR.TYPE=DARK "
                        f"from http://archive.eso.org/wdb/wdb/eso/crires/form."
                    )

                # Find bad pixel map

                file_found = False

                for key, value in self.file_dict["CAL_DARK_BPM"].items():
                    if not file_found and value["DIT"] == dit_item:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name}.fits CAL_DARK_BPM")
                        sof_open.write(f"{key} CAL_DARK_BPM\n")
                        file_found = True

                if not file_found:
                    warnings.warn(
                        f"There is not a bad pixel map with DIT = {dit_item} s."
                    )

            # Run EsoRex

            print()

            esorex = ["esorex", "cr2res_cal_flat", self.sof_file]

            if verbose:
                stdout = None
            else:
                stdout = subprocess.DEVNULL

            if not verbose:
                print("Running EsoRex...", end="", flush=True)

            subprocess.run(esorex, cwd=self.calib_folder, stdout=stdout, check=True)

            if not verbose:
                print(" [DONE]")

            # Update file dictionary with master flat

            fits_files = pathlib.Path(self.path / "calib").glob(
                "cr2res_cal_flat_*master_flat.fits"
            )

            for item in fits_files:
                self.update_files("CAL_FLAT_MASTER", str(item))

            # Update file dictionary with trace wave table

            fits_files = pathlib.Path(self.path / "calib").glob(
                "cr2res_cal_flat_*tw.fits"
            )

            for item in fits_files:
                self.update_files("CAL_FLAT_TW", str(item))

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

        self._print_section("Wavelength calibration")

        # Create SOF file

        print("Creating SOF file:")

        with open(self.sof_file, "w", encoding="utf-8") as sof_open:
            # Uranium-Neon lamp (UNE) frames

            indices = self.header_data["DPR.TYPE"] == "WAVE,UNE"

            une_found = False

            if sum(indices) == 0:
                warnings.warn("The \'raw\' folder does not contain "
                                   "any DPR.TYPE=WAVE,UNE file.")

            elif sum(indices) > 1:
                raise RuntimeError(
                    "More than one WAVE,UNE file "
                    "Please only provided a single "
                    "WAVE,UNE file."
                )

            else:
                une_found = True

            for item in self.header_data[indices]["ORIGFILE"]:
                print(f"   - raw/{item} WAVE_UNE")
                file_path = f"{self.path}/raw/{item}"
                sof_open.write(f"{file_path} WAVE_UNE\n")
                self.update_files("WAVE_UNE", file_path)

            # Fabry Pérot Etalon (FPET) frames

            indices = self.header_data["DPR.TYPE"] == "WAVE,FPET"

            fpet_found = False

            if sum(indices) == 0:
                indices = self.header_data["OBJECT"] == "WAVE,FPET"

            if sum(indices) == 0:
                warnings.warn("The \'raw\' folder does not contain "
                                   "any DPR.TYPE=WAVE,FPET file.")

            elif sum(indices) > 1:
                raise RuntimeError(
                    "More than one WAVE,FPET file "
                    "Please only provided a single "
                    "WAVE,FPET file."
                )

            else:
                fpet_found = True

            for item in self.header_data[indices]["ORIGFILE"]:
                print(f"   - raw/{item} WAVE_FPET")
                file_path = f"{self.path}/raw/{item}"
                sof_open.write(f"{file_path} WAVE_FPET\n")
                self.update_files("WAVE_FPET", file_path)

            # Find trace file

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
                for key, value in self.file_dict["CAL_FLAT_TW"].items():
                    if not file_found:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name} CAL_FLAT_TW")
                        sof_open.write(f"{key} CAL_FLAT_TW\n")
                        file_found = True

            if not file_found:
                raise RuntimeError("Could not find a trace wave table.")

            # Find emission lines file

            file_found = False

            if "EMISSION_LINES" in self.file_dict:
                for key in self.file_dict["EMISSION_LINES"]:
                    if not file_found:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name}.fits EMISSION_LINES")
                        sof_open.write(f"{key} EMISSION_LINES\n")
                        file_found = True

            if not file_found:
                raise RuntimeError("Could not find an emission lines file.")

        # Run EsoRex

        print()

        esorex = ["esorex", "cr2res_cal_wave", self.sof_file]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        if not verbose:
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=self.calib_folder, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]")

        # Update file dictionary with UNE wave tables

        if une_found:
            spec_file = f"{self.path}/calib/cr2res_cal_wave_tw_une.fits"
            self.update_files("CAL_WAVE_TW", spec_file)

            spec_file = f"{self.path}/calib/cr2res_cal_wave_wave_map_une.fits"
            self.update_files("CAL_WAVE_MAP", spec_file)

        # Update file dictionary with FPET wave tables

        if fpet_found:
            spec_file = f"{self.path}/calib/cr2res_cal_wave_tw_fpet.fits"
            self.update_files("CAL_WAVE_TW", spec_file)

            spec_file = f"{self.path}/calib/cr2res_cal_wave_wave_map_fpet.fits"
            self.update_files("CAL_WAVE_MAP", spec_file)

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

        self._print_section("Generate telluric table")

        # Create SOF file

        print("Creating SOF file:")

        with open(self.sof_file, "w", encoding="utf-8") as sof_open:
            print("   - calib/sky_spec.dat EMISSION_LINES_TXT")
            spec_file = f"{self.path}/calib/sky_spec.dat"
            sof_open.write(f"{spec_file} EMISSION_LINES_TXT\n")

        # Run EsoRex

        print()

        esorex = ["esorex", "cr2res_util_genlines", self.sof_file]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        if not verbose:
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=self.calib_folder, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]")

        # Update file dictionary with sky spectrum

        spec_file = f"{self.path}/calib/sky_spec.fits"
        self.update_files("EMISSION_LINES", spec_file)

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def util_trace(self, verbose: bool = True) -> None:
        """
        Method for running ``cr2res_util_trace``.

        Parameters
        ----------
        verbose : bool
            Print output produced by ``esorex``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section("Detect traces")

        # Create SOF file

        print("Creating SOF file:")

        indices = self.header_data["DPR.TYPE"] == "FLAT"

        with open(self.sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                print(f"   - raw/{item} FLAT")
                sof_open.write(f"{self.path}/raw/{item} FLAT\n")

        # Run EsoRex

        print()

        esorex = ["esorex", "cr2res_util_trace", self.sof_file]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        if not verbose:
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=self.calib_folder, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]")

        # Update file dictionary with trace wave table

        for item in self.header_data[indices]["ORIGFILE"]:
            trace_file = f"{self.path}/calib/{item[:-5]}_tw.fits"
            self.update_files("UTIL_TRACE_TW", trace_file)

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

        self._print_section("Process nodding frames")

        indices = self.header_data["DPR.TYPE"] == "OBJECT"

        # Check unique DIT

        unique_dit = set()
        for item in self.header_data[indices]["DET.SEQ1.DIT"]:
            unique_dit.add(item)

        print(f"Unique DIT values: {unique_dit}\n")

        # Create SOF file

        print("Creating SOF file:")

        with open(self.sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                file_path = f"{self.path}/raw/{item}"
                header = fits.getheader(file_path)

                if "ESO TPL ID" in header:
                    if header["ESO TPL ID"] == "CRIRES_spec_obs_AutoNodOnSlit":
                        print(f"   - raw/{item} CAL_NODDING_JITTER")
                        sof_open.write(f"{file_path} CAL_NODDING_JITTER\n")
                        self.update_files("CAL_NODDING_JITTER", file_path)

            # Find master flat

            file_found = False

            # for key in self.file_dict["CAL_FLAT_MASTER"]:
            #     if not file_found:
            #         file_name = key.split("/")[-1]
            #         print(f"   - calib/{file_name} CAL_FLAT_MASTER")
            #         sof_open.write(f"{key} CAL_FLAT_MASTER\n")
            #         file_found = True

            if not file_found:
                warnings.warn("Could not find a master flat.")

            # Find trace file

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
                for key, value in self.file_dict["CAL_FLAT_TW"].items():
                    if not file_found:
                        file_name = key.split("/")[-1]
                        print(f"   - calib/{file_name} CAL_FLAT_TW")
                        sof_open.write(f"{key} CAL_FLAT_TW\n")
                        file_found = True

            if not file_found:
                warnings.warn("Could not find a trace file.")

        # Run EsoRex

        print()

        esorex = ["esorex", "cr2res_obs_nodding", self.sof_file]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        if not verbose:
            print("Running EsoRex...", end="", flush=True)

        subprocess.run(esorex, cwd=self.product_folder, stdout=stdout, check=True)

        if not verbose:
            print(" [DONE]")

        # Write updated dictionary to JSON file

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

    @typechecked
    def plot_spectra(self, nod_ab: str = "A") -> None:
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

        print("Reading FITS...", end="", flush=True)

        spec_data = []

        with fits.open(fits_file) as hdu_list:
            # hdu_list.info()

            # Skip the empty PrimaryHDU
            for item in hdu_list[1:]:
                spec_data.append(item.data)

        print(" [DONE]")

        n_chip = len(spec_data)

        print("Plotting spectra...", end="", flush=True)

        for i, det_item in enumerate(spec_data):
            n_spec = len(det_item.columns) // n_chip
            plt.figure(figsize=(8, n_spec * 2))

            for j in range(n_spec):
                plt.subplot(n_spec, 1, n_spec - j)

                wavel = det_item[f"0{j+2}_01_WL"]
                flux = det_item[f"0{j+2}_01_SPEC"]
                error = det_item[f"0{j+2}_01_ERR"]

                plt.plot(wavel, flux, "-", lw=0.8, color="tab:blue")
                plt.fill_between(
                    wavel, flux - error, flux - error, color="tab:blue", alpha=0.5
                )
                plt.xlabel("Wavelength (nm)", fontsize=13)
                plt.ylabel("Flux", fontsize=13)
                plt.minorticks_on()

            plt.tight_layout()
            plt.savefig(f"{self.path}/product/spectra_{i+1}.png")
            plt.clf()

        print(" [DONE]")
