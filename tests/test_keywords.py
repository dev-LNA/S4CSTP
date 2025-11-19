# -*- coding: utf-8 -*-

"""Test the image keywords


Created on thursday, May 22 2025.

@author: denis
"""

import configparser
import inspect
import logging
import re
import unittest
from datetime import datetime, timedelta
from getpass import getuser
from os import listdir
from os.path import isdir, join
from pathlib import Path

import astropy.io.fits as fits
import numpy as np
import pandas as pd


class Test_Keywords(unittest.TestCase):
    var_types = {"float": float, "integer": int, "string": str, "boolean": bool}
    kws_specific_values = [
        "BITPIX",
        "INSTMODE",
        "SYNCMODE",
        "FILTER",
        "OBSTYPE",
        "ACQMODE",
        "PREAMP",
        "READRATE",
        "VSHIFT",
        "TRIGGER",
        "EMMODE",
        "SHUTTER",
        "TEMPST",
        "VCLKAMP",
        "CTRLINTE",
        "WPSEL",
        "CALW",
    ]
    regex_expressions = {
        "FILENAME": r"\d{8}_s4c[1-4]_\d{6}(_[a-z0-9]+)?\.fits",
        "DATE-OBS": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}",
        "UTTIME": r"\d{2}:\d{2}:\d{2}\.\d{6}",
        "UTDATE": r"\d{4}-\d{2}-\d{2}",
        "RA": r"[\+-]?\d{2}:\d{2}:\d{2}\.\d+",
        "DEC": r"[\+-]?\d{2}:\d{2}:\d{2}\.\d+",
        "TCSHA": r"[\+-]?\d{2}:\d{2}:\d{2}(\.\d+)?",
        "TCSDATE": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}",
        "ACSVRSN": r"v\d+\.\d+\.\d+",
        "GUIVRSN": r"v\d+\.\d+\.\d+",
        "ICSVRSN": r"v\d+\.\d+\.\d+",
    }
    kws_fixed_str_size = [("PROJID", 15), ("OBJECT", 30), ("OBSERVER", 54)]
    simulated_mode_kws = [
        "ACSMODE",
        "WPROMODE",
        "WPSEMODE",
        "ANMODE",
        "CALWMODE",
        "GMIRMODE",
        "GFOCMODE",
        "TCSMODE",
    ]

    @classmethod
    def setUpClass(cls):
        cfg = cls._read_config_file()
        cls.image_folder = cls._get_image_folder(cfg)
        cls.hdrs_list = cls._get_headers(cls.image_folder)
        cls.read_noises, cls.ccd_gains, cls.header_content = cls._read_csvs()

    def _read_config_file():
        sparc4_folder = join("C:\\", "Users", getuser(), "SPARC4", "ACS")
        cfg_file = join(sparc4_folder, "acs_config.cfg")
        cfg = configparser.ConfigParser()
        cfg.read(cfg_file)
        return cfg

    def _get_image_folder(cfg):
        today = Path(cfg.get("channel configuration", "image path").strip(r"\""))
        today_list = [file for file in listdir(today) if ".fits" in file]
        if today_list != []:
            return today
        yesterday = datetime.now() - timedelta(days=1)
        yesterday = join(today, "..", yesterday.strftime("%Y%m%d"))
        if not isdir(yesterday):
            raise FileNotFoundError(f"The folder {yesterday} does not exist.")
        return yesterday

    def _get_headers(image_folder):
        hdrs_list = []
        folder = join(image_folder)
        for file in listdir(folder):
            if file[-4:] != "fits":
                continue
            hdr = fits.getheader(join(folder, file))
            hdrs_list.append(hdr)
        return hdrs_list

    def _read_csvs():
        read_noises = pd.read_csv(join("csv", "read_noises.csv"))
        ccd_gains = pd.read_csv(join("csv", "preamp_gains.csv"))
        header_content = pd.read_csv(join("csv", "header_content.csv"), delimiter=";")
        return read_noises, ccd_gains, header_content

    # ----------------------------------------------------------------------------------------
    @staticmethod
    def compare_numbers(expected, received, filename, func_name):
        if not np.allclose(expected, received):
            logging.error(
                f"Test: {func_name}, filename: {filename}, expected val: {expected}, received val: {received}"
            )

    @staticmethod
    def compare_lists(expected, received, filename, func_name):
        expected, received = set(expected), set(received)
        if expected != received:
            logging.error(
                f"Test: {func_name}, filename: {filename}, diff: {expected ^ received}"
            )

    @staticmethod
    def verify_if_different(expected, received, filename, func_name):
        expected, received = set(expected), set(received)
        if expected == received:
            logging.error(
                f"Test: {func_name}, filename: {filename}, expected: {expected}, received: {received}"
            )

    @staticmethod
    def verify_type(kw, value, _type, filename, func_name):
        if not isinstance(value, _type):
            logging.error(
                f"Test: {func_name}, filename: {filename}, keyword {kw} is not an instance of {_type}: an {type(value)} was found."
            )

    @staticmethod
    def kw_in_interval(_min, _max, value, kw, filename, func_name):
        if not _min <= value <= _max:
            logging.error(
                f"Test: {func_name}, filename: {filename}, keyword {kw} is not in the interval [{_min}, {_max}]: {value}"
            )

    @staticmethod
    def val_in_list(value, _list, kw, filename, func_name):
        if not value in _list:
            logging.error(
                f"Test: {func_name}, filename: {filename}, keyword {kw} is not in {_list}: {value}"
            )

    @staticmethod
    def verify_regex(value, expression, kw, filename, func_name):
        if not re.match(expression, value):
            logging.error(
                f"Test: {func_name}, filename: {filename}, an unexpected value was found for the keyword {kw}: {value}"
            )

    @staticmethod
    def verify_str_size(value, str_size, kw, filename, func_name):
        if (n := len(value)) > str_size:
            logging.error(
                f"Test: {func_name}, filename: {filename}, the expected size for the keyword {kw} is {str_size}. However, {n} characters were found."
            )

    # -------------------------------------------------------------------------------------

    def test_missing_keywords(self):
        for hdr in self.hdrs_list:
            if "COMMENT" in hdr.keys():
                del hdr["COMMENT"]
            hdr_keywords = list(hdr.keys())
            csv_keywords = list(self.header_content["Keyword"].values)
            func_name = inspect.currentframe().f_code.co_name
            self.compare_lists(csv_keywords, hdr_keywords, hdr["FILENAME"], func_name)

    def test_kw_comments(self):
        for hdr in self.hdrs_list:
            if "COMMENT" in hdr.keys():
                del hdr["COMMENT"]
            hdr_comment = hdr.comments
            csv_comment = self.header_content["Comment"]
            func_name = inspect.currentframe().f_code.co_name
            self.compare_lists(csv_comment, hdr_comment, hdr["FILENAME"], func_name)

    def test_keywords_types(self):
        func_name = inspect.currentframe().f_code.co_name
        for hdr in self.hdrs_list:
            for _, row in self.header_content.iterrows():
                kw = row["Keyword"]
                try:
                    keyword_val = hdr[kw]
                    type = self.var_types[row["Type"]]
                    self.verify_type(kw, keyword_val, type, hdr["FILENAME"], func_name)
                except Exception as e:
                    logging.error(
                        f"Test: {func_name}, filename: {hdr["FILENAME"]}, keyword: {kw}, {repr(e)}"
                    )
        return

    def test_kws_in_interval(self):
        func_name = inspect.currentframe().f_code.co_name
        filtered_hdr_content = self.header_content[
            self.header_content["Type"].isin(["integer", "float"])
        ]
        for hdr in self.hdrs_list:
            for _, row in filtered_hdr_content.iterrows():
                try:
                    keyword = row["Keyword"]
                    value = hdr[keyword]
                    filename = hdr["FILENAME"]
                    if keyword not in self.kws_specific_values:
                        _min, _max = row["Allowed values"].split(",")
                        _min = float(_min)
                        if _max == "inf":
                            _max = np.inf
                        else:
                            _max = float(_max)

                        self.kw_in_interval(
                            _min, _max, value, keyword, filename, func_name
                        )
                except Exception as e:
                    logging.error(
                        f"Test: {func_name}, filename: {hdr["FILENAME"]}, {repr(e)}"
                    )
        return

    def test_kws_specific_vals(self):
        for hdr in self.hdrs_list:
            for kw in self.kws_specific_values:
                row = self.header_content[self.header_content["Keyword"] == kw]
                allowed_vals = row["Allowed values"].values[0].split(",")
                _type = row["Type"].values[0]
                if _type in ["integer", "float"]:
                    allowed_vals = [
                        self.var_types[_type](new_val) for new_val in allowed_vals
                    ]
                file_name = hdr["FILENAME"]
                func_name = inspect.currentframe().f_code.co_name
                value = hdr[kw]
                self.val_in_list(value, allowed_vals, kw, file_name, func_name)
                assert hdr[kw] in allowed_vals

    def test_kws_regex(self):
        for hdr in self.hdrs_list:
            for kw in self.regex_expressions:
                expression = self.regex_expressions[kw]
                value = hdr[kw]
                filename = hdr["FILENAME"]
                func_name = inspect.currentframe().f_code.co_name
                self.verify_regex(value, expression, kw, filename, func_name)
        return

    def test_WPPOS(self):
        for hdr in self.hdrs_list:
            if (hdr["INSTMODE"] == "POLAR") & (hdr["WPPOS"] == 0):
                raise ValueError(
                    f"The value WPPOS=0 was found the polarimetric mode.",
                    hdr["FILENAME"],
                )
        return

    def test_comment_kw(self):
        for hdr in self.hdrs_list:
            if "COMMENT" in hdr.keys():
                filename = hdr["FILENAME"]
                expected = ""
                received = hdr["COMMENT"]
                func_name = inspect.currentframe().f_code.co_name
                self.verify_if_different(expected, received, filename, func_name)

    # -------------------- tests to verify the keywords content ----------------------------

    def test_observatory_coords(self):
        for hdr in self.hdrs_list:
            filename = hdr["FILENAME"]
            func_name = inspect.currentframe().f_code.co_name

            received = hdr["OBSLONG"]
            expected = -45.5825
            self.compare_numbers(expected, received, filename, func_name)

            received = hdr["OBSLAT"]
            expected = -22.534
            self.compare_numbers(expected, received, filename, func_name)

            received = hdr["OBSALT"]
            expected = 1864.0
            self.compare_numbers(expected, received, filename, func_name)

    def test_ccd_gain(self):
        for hdr in self.hdrs_list:
            em_mode = hdr["EMMODE"]
            if em_mode != "Conventional":
                em_mode = "EM"
            readout = hdr["READRATE"]
            preamp = float(hdr["PREAMP"][-1])
            serial_number = f'{hdr["CCDSERN"]}'
            gain = hdr["GAIN"]
            filter = (
                (self.ccd_gains["EM Mode"] == em_mode)
                & (self.ccd_gains["Readout Rate"] == readout)
                & (self.ccd_gains["Preamp"] == preamp)
            )
            line = self.ccd_gains[filter]
            filename, expected, received = (
                hdr["FILENAME"],
                line[serial_number].values[0],
                gain,
            )
            func_name = inspect.currentframe().f_code.co_name
            self.compare_numbers(expected, received, filename, func_name)

    def test_read_noise(self):
        for hdr in self.hdrs_list:
            em_mode = hdr["EMMODE"]
            if em_mode != "Conventional":
                em_mode = "EM"
            readout = hdr["READRATE"]
            preamp = float(hdr["PREAMP"][-1])
            serial_number = f'{hdr["CCDSERN"]}'
            read_noise = hdr["RDNOISE"]
            filter = (
                (self.read_noises["EM Mode"] == em_mode)
                & (self.read_noises["Readout Rate"] == readout)
                & (self.read_noises["Preamp"] == preamp)
            )
            line = self.read_noises[filter]
            filename, expected, received = (
                hdr["FILENAME"],
                line[serial_number].values[0],
                read_noise,
            )
            func_name = inspect.currentframe().f_code.co_name
            self.compare_numbers(expected, received, filename, func_name)

    def test_equinox(self):
        for hdr in self.hdrs_list:
            filename, expected, received = (hdr["FILENAME"], 2000.0, hdr["EQUINOX"])
            func_name = inspect.currentframe().f_code.co_name
            self.compare_numbers(expected, received, filename, func_name)

    def test_BSCALE(self):
        for hdr in self.hdrs_list:
            filename, expected, received = (hdr["FILENAME"], 1, hdr["BSCALE"])
            func_name = inspect.currentframe().f_code.co_name
            self.compare_numbers(expected, received, filename, func_name)

    def test_BZERO(self):
        for hdr in self.hdrs_list:
            received = hdr["BZERO"]
            expected = 2**15
            filename = hdr["FILENAME"]
            func_name = inspect.currentframe().f_code.co_name
            self.compare_numbers(expected, received, filename, func_name)

    def test_BITPIX(self):
        for hdr in self.hdrs_list:
            filename, expected, received = (hdr["FILENAME"], 16, hdr["BITPIX"])
            func_name = inspect.currentframe().f_code.co_name
            self.compare_numbers(expected, received, filename, func_name)

    def test_NAXIS(self):
        for hdr in self.hdrs_list:
            filename, expected, received = (hdr["FILENAME"], 2, hdr["NAXIS"])
            func_name = inspect.currentframe().f_code.co_name
            self.compare_numbers(expected, received, filename, func_name)

    def test_kw_sizes(self):
        for hdr in self.hdrs_list:
            for kw, str_size in self.kws_fixed_str_size:
                kw_value, filename = hdr[kw], hdr["FILENAME"]
                func_name = inspect.currentframe().f_code.co_name
                self.verify_str_size(kw_value, str_size, kw, filename, func_name)

    def test_simulated_mode(self):
        func_name = inspect.currentframe().f_code.co_name
        for hdr in self.hdrs_list:
            for kw in self.simulated_mode_kws:
                if hdr[kw] == False:
                    logging.error(
                        f"Test: {func_name}, filename: {hdr['FILENAME']}, the keyword {kw} was set in the simulated mode."
                    )

    def test_empty_observer_kw(self):
        func_name = inspect.currentframe().f_code.co_name
        for hdr in self.hdrs_list:
            if hdr["OBSERVER"] == "":
                logging.error(
                    f"Test: {func_name}, filename: {hdr['FILENAME']}, the keyword OBSERVER is empty."
                )
