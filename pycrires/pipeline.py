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

        self._print_section("Pipeline for VLT/CRIRES", bound_char="=", extra_line=False)

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
            if item.suffix == ".fits":
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
                        f"The ARCFILE keyword is not found in "
                        f"the header of {item.name}"
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
            if file_item.suffix == ".fits":
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
            Partial filename.

        Returns
        -------
        NoneType
            None
        """

        if sof_key in self.file_dict:
            if f"{self.path}/{file_name}" not in self.file_dict[sof_key]:
                self.file_dict[sof_key] += [f"{self.path}/{file_name}"]
        else:
            self.file_dict[sof_key] = [f"{self.path}/{file_name}"]

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

        sky_calc = skycalc_ipy.SkyCalc()
        sky_spec = sky_calc.get_sky_spectrum(filename=temp_file)

        emission_spec = np.column_stack(
            (1e3 * sky_spec["lam"], 1e-3 * sky_spec["flux"])
        )
        header = "Wavelength (nm) - Flux (ph arcsec-2 m-2 s-1 nm-1)"

        print("Storing emission spectrum: calib/sky_spec.dat")
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

        print("Creating .sof file:")

        indices = self.header_data["DPR.TYPE"] == "DARK"

        with open(self.sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                print(f"   - raw/{item} DARK")
                sof_open.write(f"{self.path}/raw/{item} DARK\n")

                self.update_files("DARK", f"calib/{item}")

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

        esorex = ["esorex", "cr2res_cal_dark", self.sof_file]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        subprocess.run(esorex, cwd=self.calib_folder, stdout=stdout, check=True)

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

        print("Creating .sof file:")

        indices = self.header_data["DPR.TYPE"] == "FLAT"

        with open(self.sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                print(f"   - raw/{item} FLAT")
                sof_open.write(f"{self.path}/raw/{item} FLAT\n")
                self.update_files("FLAT", f"calib/{item}")

            print("   - calib/cr2res_cal_dark_master.fits CAL_DARK_MASTER")
            sof_open.write(
                f"{self.path}/calib/cr2res_cal_dark_master.fits CAL_DARK_MASTER\n"
            )

            print("   - calib/cr2res_cal_dark_bpm.fits CAL_DARK_BPM")
            sof_open.write(f"{self.path}/calib/cr2res_cal_dark_bpm.fits CAL_DARK_BPM\n")

            self.update_files("CAL_DARK_MASTER", "calib/cr2res_cal_dark_master.fits")
            self.update_files("CAL_DARK_BPM", "calib/cr2res_cal_dark_bpm.fits")

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

        esorex = ["esorex", "cr2res_cal_flat", self.sof_file]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        subprocess.run(esorex, cwd=self.calib_folder, stdout=stdout, check=True)

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

        print("Creating .sof file:")

        with open(self.sof_file, "w", encoding="utf-8") as sof_open:
            indices = self.header_data["DPR.TYPE"] == "WAVE,UNE"
            for item in self.header_data[indices]["ORIGFILE"]:
                print(f"   - raw/{item} WAVE_UNE")
                sof_open.write(f"{self.path}/raw/{item} WAVE_UNE\n")
                self.update_files("WAVE_UNE", f"raw/{item}")

            indices = self.header_data["DPR.TYPE"] == "WAVE,FPET"
            for item in self.header_data[indices]["ORIGFILE"]:
                print(f"   - raw/{item} WAVE_FPET")
                sof_open.write(f"{self.path}/raw/{item} WAVE_FPET\n")
                self.update_files("WAVE_FPET", f"raw/{item}")

            print("   - calib/cr2res_cal_dark_master.fits CAL_DARK_MASTER")
            sof_open.write(
                f"{self.path}/calib/cr2res_cal_dark_master.fits CAL_DARK_MASTER\n"
            )

            print("   - calib/cr2res_cal_dark_bpm.fits CAL_DARK_BPM")
            sof_open.write(f"{self.path}/calib/cr2res_cal_dark_bpm.fits CAL_DARK_BPM\n")

            print("   - calib/cr2res_cal_flat_Open_master_flat.fits CAL_FLAT_MASTER")
            sof_open.write(
                f"{self.path}/calib/cr2res_cal_flat_Open_master_flat.fits "
                "CAL_FLAT_MASTER\n"
            )

            print("   - calib/cr2res_cal_flat_Open_tw.fits CAL_FLAT_TW")
            sof_open.write(
                f"{self.path}/calib/cr2res_cal_flat_Open_tw.fits CAL_FLAT_TW\n"
            )

            print("   - calib/sky_spec.fits EMISSION_LINES")
            sof_open.write(f"{self.path}/calib/sky_spec.fits EMISSION_LINES\n")

            self.update_files("CAL_DARK_MASTER", "calib/cr2res_cal_dark_master.fits")
            self.update_files("CAL_DARK_BPM", "calib/cr2res_cal_dark_bpm.fits")
            self.update_files(
                "CAL_FLAT_MASTER", "calib/cr2res_cal_flat_Open_master_flat.fits"
            )
            self.update_files("CAL_FLAT_TW", "calib/cr2res_cal_flat_Open_tw.fits")
            self.update_files("EMISSION_LINES_TXT", "calib/sky_spec.fits")

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

        esorex = ["esorex", "cr2res_cal_wave", self.sof_file]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        subprocess.run(esorex, cwd=self.calib_folder, stdout=stdout, check=True)

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

        self._print_section("Generate calibration lines")

        print("Creating .sof file:")

        with open(self.sof_file, "w", encoding="utf-8") as sof_open:
            print("   - calib/sky_spec.dat EMISSION_LINES_TXT")
            sof_open.write(f"{self.path}/calib/sky_spec.dat EMISSION_LINES_TXT\n")

        self.update_files("EMISSION_LINES_TXT", "calib/sky_spec.fits")

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

        esorex = ["esorex", "cr2res_util_genlines", self.sof_file]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        subprocess.run(esorex, cwd=self.calib_folder, stdout=stdout, check=True)

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

        print("Creating .sof file:")

        indices = self.header_data["DPR.TYPE"] == "FLAT"

        with open(self.sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                print(f"   - raw/{item} FLAT")
                sof_open.write(f"{self.path}/raw/{item} FLAT\n")

                if "FLAT" in self.file_dict:
                    self.file_dict["FLAT"] += [f"{self.path}/raw/{item}"]
                else:
                    self.file_dict["FLAT"] = [f"{self.path}/raw/{item}"]

                if "UTIL_TRACE_TW" in self.file_dict:
                    self.file_dict["UTIL_TRACE_TW"] += [
                        f"{self.path}/calib/{item[:-5]}_tw.fits"
                    ]
                else:
                    self.file_dict["UTIL_TRACE_TW"] = [
                        f"{self.path}/calib/{item[:-5]}_tw.fits"
                    ]

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

        esorex = ["esorex", "cr2res_util_trace", self.sof_file]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        subprocess.run(esorex, cwd=self.calib_folder, stdout=stdout, check=True)

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

        self._print_section("Process nodding observation")

        print("Creating .sof file:")

        indices = self.header_data["TPL.ID"] == "CRIRES_spec_obs_AutoNodOnSlit"

        with open(self.sof_file, "w", encoding="utf-8") as sof_open:
            for item in self.header_data[indices]["ORIGFILE"]:
                print(f"   - raw/{item} CAL_NODDING_JITTER")
                sof_open.write(f"{self.path}/raw/{item} CAL_NODDING_JITTER\n")
                self.update_files("CAL_NODDING_JITTER", f"raw/{item}")

            print("   - calib/cr2res_cal_dark_master.fits CAL_DARK_MASTER")
            sof_open.write(
                f"{self.path}/calib/cr2res_cal_dark_master.fits CAL_DARK_MASTER\n"
            )

            print("   - calib/cr2res_cal_dark_bpm.fits CAL_DARK_BPM")
            sof_open.write(f"{self.path}/calib/cr2res_cal_dark_bpm.fits CAL_DARK_BPM\n")

            print("   - calib/cr2res_cal_flat_Open_master_flat.fits CAL_FLAT_MASTER")
            sof_open.write(
                f"{self.path}/calib/cr2res_cal_flat_Open_master_flat.fits "
                "CAL_FLAT_MASTER\n"
            )

            print("   - calib/cr2res_cal_flat_Open_tw.fits CAL_FLAT_TW")
            sof_open.write(
                f"{self.path}/calib/cr2res_cal_flat_Open_tw.fits CAL_FLAT_TW\n"
            )

            # for item in self.file_dict["UTIL_TRACE_TW"]:
            #     filename = item.split("/")[-1]
            #     print(f"   - calib/{filename} UTIL_TRACE_TW")
            #     sof_open.write(f"{item} UTIL_TRACE_TW\n")

            self.update_files("CAL_DARK_MASTER", "calib/cr2res_cal_dark_master.fits")
            self.update_files("CAL_DARK_BPM", "calib/cr2res_cal_dark_bpm.fits")
            self.update_files(
                "CAL_FLAT_MASTER", "calib/cr2res_cal_flat_Open_master_flat.fits"
            )
            self.update_files("CAL_FLAT_TW", "calib/cr2res_cal_flat_Open_tw.fits")

        with open(self.json_file, "w", encoding="utf-8") as json_file:
            json.dump(self.file_dict, json_file, indent=4)

        esorex = ["esorex", "cr2res_obs_nodding", self.sof_file]

        if verbose:
            stdout = None
        else:
            stdout = subprocess.DEVNULL

        subprocess.run(esorex, cwd=self.product_folder, stdout=stdout, check=True)
