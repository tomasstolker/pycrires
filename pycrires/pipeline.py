"""
Module with the main pipeline class.
"""

import logging
import os
import pathlib
import warnings

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.io import fits
from typeguard import typechecked


log_book = logging.getLogger(__name__)


class Pipeline:
    """
    Class for the data reduction pipeline.
    """

    @typechecked
    def __init__(self,
                 path: str) -> None:
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

        self._print_section('Pipeline for VLT/CRIRES',
                            bound_char='=',
                            extra_line=False)

        # Absolute path of the main reduction folder
        self.path = pathlib.Path(path).resolve()

        print(f'Data reduction folder: {self.path}')

        self.header_file = pathlib.Path(self.path/'header.csv')
        self.excel_file = pathlib.Path(self.path/'header.xlsx')

        if self.header_file.is_file():
            self.header_data = pd.read_csv(self.header_file)
            print('Reading header data from header.csv')

        else:
            self.header_data = pd.DataFrame()
            print('Creating header DataFrame')

    @staticmethod
    @typechecked
    def _print_section(sect_title: str,
                       bound_char: str = '-',
                       extra_line: bool = True) -> None:
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
            print('\n'+len(sect_title)*bound_char)
        else:
            print(len(sect_title)*bound_char)

        print(sect_title)
        print(len(sect_title)*bound_char+'\n')

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

        self._print_section('Observation details')

        check_key = {'OBS.TARG.NAME': 'Target',
                     'OBS.PROG.ID': 'Program ID',
                     'INS.WLEN.ID': 'Wavelength setting',
                     'INS.WLEN.CWLEN': 'Central wavelength (nm)',
                     'INS1.DROT.POSANG': 'Position angle (deg)',
                     'INS.SLIT1.WID': 'Slit width (arcsec)',
                     'INS.GRAT1.ORDER': 'Grating order'}

        if 'RA' in self.header_data and 'DEC' in self.header_data:
            ra_mean = np.mean(self.header_data['RA'])
            dec_mean = np.mean(self.header_data['DEC'])

            target_coord = SkyCoord(ra_mean, dec_mean,
                                    unit='deg', frame='icrs')

            ra_dec = target_coord.to_string('hmsdms')

            print(f'RA Dec = {ra_dec}')

        for key, value in check_key.items():
            header = self.header_data[key].to_numpy()

            if isinstance(header[0], str):
                indices = np.where(header is not None)[0]

            else:
                indices = ~np.isnan(header)
                indices[header == 0.] = False

            if np.all(header[indices] == header[indices][0]):
                print(f'{value} = {header[0]}')

            else:
                warnings.warn(f'Expecting a single value for {key} but '
                              f'multiple values are found: {header}')

                if isinstance(header[indices][0], np.float64):
                    print(f'{value} = {np.mean(header)}')

        if 'OBS.ID' in self.header_data:
            # obs_id = self.header_data['OBS.ID']
            unique_id = pd.unique(self.header_data['OBS.ID'])

            print('\nObservation ID:')

            for item in unique_id:
                count_files = np.sum(self.header_data['OBS.ID'] == item)
                print(f'   - {item} -> {count_files} files')

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

        self.header_data.sort_values(['DET.EXP.ID'],
                                     ascending=True,
                                     inplace=True)

        # Write DataFrame to CSV file

        print(f'Exporting DataFrame to {self.header_file.name}')

        self.header_data.to_csv(self.header_file,
                                sep=',',
                                header=True,
                                index=False)

        # Write DataFrame to Excel file

        print(f'Exporting DataFrame to {self.excel_file.name}')

        self.header_data.to_excel(self.excel_file,
                                  sheet_name='CRIRES',
                                  header=True,
                                  index=False)

    @typechecked
    def rename_files(self) -> None:
        """
        Method for renaming the files from ``ARCFILE`` to ``ORIGFILE``.

        Returns
        -------
        NoneType
            None
        """

        self._print_section('Renaming files')

        raw_files = sorted(pathlib.Path(self.path/'raw').glob('*.fits'))

        n_total = 0
        n_renamed = 0

        acq_files = []
        science_files = []
        calib_files = []

        for item in raw_files:
            if item.suffix == '.fits':
                header = fits.getheader(item)

                if 'ESO DPR CATG' in header:
                    dpr_catg = header['ESO DPR CATG']

                    if dpr_catg == 'SCIENCE':
                        science_files.append(item)

                    elif dpr_catg in 'CALIB':
                        calib_files.append(item)

                    elif dpr_catg == 'ACQUISITION':
                        acq_files.append(item)

                    else:
                        warnings.warn(f'The DPR.CATG with value {dpr_catg} '
                                      f'has not been recognized.')

                if 'ARCFILE' in header and 'ORIGFILE' in header:
                    if item.name == header['ARCFILE']:
                        os.rename(item, item.parent/header['ORIGFILE'])
                        n_renamed += 1

                elif 'ARCFILE' not in header:
                    warnings.warn(f'The ARCFILE keyword is not found in '
                                  f'the header of {item.name}')

                elif 'ORIGFILE' not in header:
                    warnings.warn(f'The ORIGFILE keyword is not found in '
                                  f'the header of {item.name}')

            n_total += 1

        print('Science data:\n')
        for item in science_files:
            print(f'   - {item.name}')

        print('\nCalibration data:\n')
        for item in calib_files:
            print(f'   - {item.name}')

        print('\nAcquisition data:\n')
        for item in acq_files:
            print(f'   - {item.name}')

        print(f'\nTotal tumber of FITS files: {n_total}')
        print(f'Number of renamed files: {n_renamed}')

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

        self._print_section('Extracting FITS headers')

        # Create a new DataFrame
        self.header_data = pd.DataFrame()
        print('Creating new DataFrame...\n')

        key_file = os.path.dirname(__file__) + '/keywords.txt'
        keywords = np.genfromtxt(key_file, dtype='str', delimiter=',')

        raw_files = pathlib.Path(self.path/'raw').glob('*.fits')

        header_dict = {}
        for key_item in keywords:
            header_dict[key_item] = []

        for file_item in raw_files:
            if file_item.suffix == '.fits':
                header = fits.getheader(file_item)

                for key_item in keywords:
                    if key_item in header:
                        header_dict[key_item].append(header[key_item])
                    else:
                        header_dict[key_item].append(None)

        for key_item in keywords:
            column_name = key_item.replace(' ', '.')
            column_name = column_name.replace('ESO.', '')

            self.header_data[column_name] = header_dict[key_item]

        self._export_header()
        self._print_info()
