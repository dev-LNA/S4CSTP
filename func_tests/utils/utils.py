import configparser
import logging
import subprocess
from pathlib import Path
from time import sleep
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


def read_config_file(file_name: str = "acs_config.cfg") -> data_types.S4ACS_Cfg_File:
    section_name = "channel configuration"
    cfg_file_folder = Path.home() / "SPARC4" / "ACS"
    cfg_file = cfg_file_folder / file_name
    cfg_file_content = {}
    if not cfg_file.exists():
        raise RuntimeError(f"file {cfg_file} not found")
    config = configparser.ConfigParser()
    config.read(cfg_file)

    cfg_file_content = data_types.S4ACS_Cfg_File(
        channel=int(config.get(section_name, "channel")),
        acs_mode=config.get(section_name, "ACS mode") == 1,
        image_path=Path(config.get(section_name, "image path")),
        log_file_path=Path(config.get(section_name, "log file path")),
        log_level=data_types.Log_Level(
            _log_levels[config.get(section_name, "log level")]
        ),
    )
    return cfg_file_content


def write_cfg_file(
    new_cfg: data_types.S4ACS_Cfg_File, file_name: str = "acs_config.cfg"
) -> None:
    cfg_file_folder = Path.home() / "SPARC4" / "ACS"
    cfg_file = cfg_file_folder / file_name
    config = configparser.ConfigParser()
    config["channel configuration"] = new_cfg.to_sparc4_format()
    with open(cfg_file, "w") as file:
        config.write(file)

    return


def run_s4acs_exe() -> None:
    import pyautogui

    logging.info("Starting S4ACS executable...")
    window = _is_window_open("S4ACS.vi")
    if window is None:
        exe = Path.home() / "Desktop" / "S4ACSv1.56.2" / "S4ACS.exe"
        subprocess.Popen([str(exe)])
    else:
        window.set_focus()
    sleep(2)
    pyautogui.hotkey("ctrl", "r")


def _is_window_open(title: str) -> Any | None:
    from pywinauto import Desktop

    for window in Desktop(backend="uia").windows():
        if title in window.window_text():
            return window
    return None
