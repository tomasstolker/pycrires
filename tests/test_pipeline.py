import os
import shutil
import tarfile

import pooch
import pytest

import pycrires


class TestPipeline:
    def setup_class(self) -> None:

        self.test_dir = os.path.dirname(__file__) + "/"

        data_folder = self.test_dir + "raw/"

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        url = "https://home.strw.leidenuniv.nl/~stolker/pycrires/betapic_crires.tgz"
        data_file = self.test_dir + "raw/betapic_crires.tgz"

        pooch.retrieve(
            url=url,
            known_hash="a3dd312f1d8115b1ce0eee5929f1287ef5764f82d649731aa859a11b1a545fa6",
            fname="betapic_crires.tgz",
            path=os.path.join(self.test_dir, "raw"),
            progressbar=True,
        )

        with tarfile.open(data_file) as open_tar:
            def is_within_directory(directory, target):

                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)

                prefix = os.path.commonprefix([abs_directory, abs_target])

                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(open_tar, data_folder)

        self.limit = 1e-8
        self.pipeline = pycrires.Pipeline(self.test_dir)

        self.esorex_error = "Esorex is not accessible from the " \
                            "command line. Please make sure that " \
                            "the ESO pipeline is correctly " \
                            "installed and included in the PATH " \
                            "variable."

    def teardown_class(self) -> None:

        if os.path.exists(self.test_dir + "header.csv"):
            os.remove(self.test_dir + "header.csv")

        if os.path.exists(self.test_dir + "header.xlsx"):
            os.remove(self.test_dir + "header.xlsx")

        if os.path.exists(self.test_dir + "files.json"):
            os.remove(self.test_dir + "files.json")

        shutil.rmtree(self.test_dir + "calib")
        shutil.rmtree(self.test_dir + "config")
        shutil.rmtree(self.test_dir + "product")
        shutil.rmtree(self.test_dir + "raw")

    def test_pipeline(self) -> None:

        assert isinstance(self.pipeline, pycrires.pipeline.Pipeline)

    def test_rename_files(self) -> None:

        self.pipeline.rename_files()

    def test_extract_header(self) -> None:

        self.pipeline.extract_header()

    def test_cal_dark(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.cal_dark(verbose=False)

                assert str(error.value) == self.esorex_error

        else:
            self.pipeline.cal_dark(verbose=False)

    def test_util_calib_flat(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_calib(calib_type="flat", verbose=False)

                assert str(error.value) == self.esorex_error

        else:
            self.pipeline.util_calib(calib_type="flat", verbose=False)

    def test_util_trace(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_trace(plot_trace=False, verbose=False)

                assert str(error.value) == self.esorex_error

        else:
            self.pipeline.util_trace(plot_trace=False, verbose=False)

    def test_util_slit_curv(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_slit_curv(plot_trace=True, verbose=False)

                assert str(error.value) == "The UTIL_TRACE_TW file is not found " \
                                           "in the 'calib' folder. Please first " \
                                           "run the util_trace method."

        else:
            self.pipeline.util_slit_curv(plot_trace=True, verbose=False)

    def test_util_extract_flat(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_extract(calib_type="flat", verbose=False)

                assert str(error.value) == "The UTIL_CALIB file is not found in " \
                                           "the 'calib' folder. Please first " \
                                           "run the util_calib method."

        else:
            self.pipeline.util_extract(calib_type="flat", verbose=False)

    def test_util_normflat(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_normflat(verbose=False)

                assert str(error.value) == "The UTIL_CALIB file is not found in " \
                                           "the 'calib' folder. Please first " \
                                           "run the util_calib method."

        else:
            self.pipeline.util_normflat(verbose=False)

    def test_util_calib_une(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_calib(calib_type="une", verbose=False)

                assert str(error.value) == self.esorex_error

        else:
            self.pipeline.util_calib(calib_type="une", verbose=False)

    def test_util_extract_une(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_extract(calib_type="une", verbose=False)

                assert str(error.value) == "The UTIL_CALIB file is not found in " \
                                           "the 'calib' folder. Please first " \
                                           "run the util_calib method."

        else:
            self.pipeline.util_extract(calib_type="une", verbose=False)

    def test_util_genlines(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_genlines(verbose=False)

                assert str(error.value) == self.esorex_error

        else:
            self.pipeline.util_genlines(verbose=False)

    def test_util_wave_une(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_wave(calib_type="une", verbose=False)

                assert str(error.value) == "The EMISSION_LINES file is not " \
                                           "found in the 'calib/genlines' " \
                                           "folder. Please first run the " \
                                           "util_genlines method."

        else:
            self.pipeline.util_wave(calib_type="une", poly_deg=0, wl_err=0.1, verbose=False)
            self.pipeline.util_wave(calib_type="une", poly_deg=2, wl_err=0.03, verbose=False)

    def test_util_calib_fpet(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_calib(calib_type="fpet", verbose=False)

                assert str(error.value) == self.esorex_error

        else:
            self.pipeline.util_calib(calib_type="fpet", verbose=False)

    def test_util_extract_fpet(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_extract(calib_type="fpet", verbose=False)

                assert str(error.value) == "The UTIL_CALIB file is not found in " \
                                           "the 'calib' folder. Please first " \
                                           "run the util_calib method."

        else:
            self.pipeline.util_extract(calib_type="fpet", verbose=False)

    def test_util_wave_fpet(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.util_wave(calib_type="fpet", verbose=False)

                assert str(error.value) == "The UTIL_EXTRACT_1D file is not " \
                                           "found in the 'calib/" \
                                           "util_extract_fpet' folder. Please " \
                                           "first run the util_extract method " \
                                           "with calib_type='une'."

        else:
            self.pipeline.util_wave(calib_type="fpet", poly_deg=4, wl_err=0.01, verbose=False)

    def test_obs_nodding(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(RuntimeError) as error:
                self.pipeline.obs_nodding(verbose=False, correct_bad_pixels=False, extraction_required=False)

        else:
            self.pipeline.obs_nodding(verbose=False, correct_bad_pixels=False, extraction_required=False)

    def test_run_skycalc(self) -> None:

        self.pipeline.run_skycalc(pwv=1.0)

    def test_plot_spectra(self) -> None:

        if shutil.which("esorex") is None:
            with pytest.raises(FileNotFoundError):
                self.pipeline.plot_spectra(nod_ab="A", telluric=True, corrected=False, file_id=0)

        else:
            self.pipeline.plot_spectra(nod_ab="A", telluric=True, corrected=False, file_id=0)
