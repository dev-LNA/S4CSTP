import configparser
import getpass
import logging
from pathlib import Path
from typing import Any

import func_tests.data_types as data_types

default_cam_config = {
    "EM_MODE": 1,
    "EM_GAIN": 2,
    "FRAME_TRANSFER": False,
    "READOUT_RATE": 0,
    "PREAMP": 0,
    "READ_MODE": 4,
    "ACQUISITION_MODE": 3,
    "TRIGGER_MODE": 0,
    "VERTICAL_CLOCK_VOLTAGE": 0,
    "VERTICAL_SHIFT_SPEED": 3,
    "SHUTTER_MODE": 2,
    "SHUTTER_TTL": 0,
    "SHUTTER_OPENING_TIME": 50,
    "SHUTTER_CLOSING_TIME": 50,
    "INITIAL_COLUMN": 1,
    "INITIAL_LINE": 1,
    "FINAL_COLUMN": 1024,
    "FINAL_LINE": 1024,
    "VBIN": 1,
    "HBIN": 1,
    "AD_CHANNEL": 0,
}
default_acq_config = {
    "EXPTIME": 2,
    "#FRAMES": 1,
    "#CYCLES": 1,
    "suffix": "",
    "COOLER_POWER_STATUS": 1,
    "TEMP": 20,
    "WAVEPLATE_POS": 1,
}

_log_levels = {
    "0": "STATUS",
    "1": logging.DEBUG,
    "2": logging.INFO,
    "3": logging.WARNING,
    "4": logging.ERROR,
    "5": logging.CRITICAL,
}


def read_config_file() -> dict[str, Any]:
    section_name = "channel configuration"
    cfg_file_folder = Path(f"C:/Users/{getpass.getuser()}/SPARC4/ACS")
    cfg_file = cfg_file_folder / "acs_config.cfg"
    cfg_file_content = {}
    if not cfg_file.exists():
        raise RuntimeError(f"file {cfg_file} not found")
    config = configparser.ConfigParser()
    config.read(cfg_file)
    cfg_file_content["channel"] = config.get(section_name, "channel")
    cfg_file_content["acs_mode"] = config.get(section_name, "s4acs mode") == 1
    cfg_file_content["log_level"] = data_types.Log_Level(
        _log_levels[config.get(section_name, "log level")]
    )
    log_folder = config.get(section_name, "log file path")
    cfg_file_content["log_folder"] = Path(log_folder.replace('"', ""))
    imgs_folder = config.get(section_name, "image path")
    cfg_file_content["imgs_folder"] = Path(imgs_folder.replace('"', ""))
    return cfg_file_content
