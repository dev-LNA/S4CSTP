import configparser
import json
import logging
import subprocess
from pathlib import Path
from time import sleep
from typing import Any, Final

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


S4GUI_JSON: Final = {
    "OBJECT": "Test",
    "OBSERVER": "Denis",
    "CTRLINTE": "S4GEI",
    "PROJID": "ENG",
    "SYNCMODE": "ASYNC",
    "INSTMODE": "PHOT",
    "FILTER": "CLEAR",
    "OBSTYPE": "OBJECT",
    "GUIVRSN": "v0.0.0",
    "COMMENT": "",
    "broker": "S4GEI",
    "timestamp": "0000-00-00T00:00:00.0",
    "CHANNEL 1": True,
    "CHANNEL 2": False,
    "CHANNEL 3": False,
    "CHANNEL 4": False,
    "TCSMODE": True,
}

_2nd_PART_S4ICS_PUB = {
    "broker": "S4ICS",
    "version": "v0.0.0",
    "comment": "this is a comment",
    "tcpServerSocket": "192.168.1.170",
    "tcpServerEnabled": True,
    "timestamp": "0000-00-00T00:0:000.0",
    "mechanisms": [
        {
            "name": "WPROT",
            "status": {
                "mode": "ACTIVE",
                "condition": "READY",
                "position": "0",
                "pos_name": "HOME",
                "pos_id": "-1",
            },
        },
        {
            "name": "WPSEL",
            "status": {
                "mode": "ACTIVE",
                "condition": "READY",
                "position": "50",
                "pos_name": "OFF",
                "pos_id": "2",
            },
        },
        {
            "name": "CALW",
            "status": {
                "mode": "ACTIVE",
                "condition": "READY",
                "position": "144",
                "pos_name": "OFF",
                "pos_id": "3",
            },
        },
        {
            "name": "ASEL",
            "status": {
                "mode": "ACTIVE",
                "condition": "READY",
                "position": "0",
                "pos_name": "OFF",
                "pos_id": "1",
            },
        },
        {
            "name": "GMIR",
            "status": {
                "mode": "ACTIVE",
                "condition": "READY",
                "position": "0",
                "pos_name": "HOME",
                "pos_id": "0",
            },
        },
        {
            "name": "GFOC",
            "status": {
                "mode": "ACTIVE",
                "condition": "READY",
                "position": "0",
                "pos_name": "HOME",
                "pos_id": "0",
            },
        },
    ],
}
S4ICS_JSON: Final = (
    "WPROT SIMULATED READY 0.000 NONE -1, WPSEL SIMULATED READY 80.000 OFF 2, CALW SIMULATED READY 216.000 SHUTTER 4, ASEL SIMULATED READY 0.000 OFF 1, GMIR SIMULATED READY 0.000 TARGET 1, GFOC SIMULATED READY 0.000 TARGET 1\n"
    + json.dumps(_2nd_PART_S4ICS_PUB)
)

WEATHER_JSON: Final = {
    "broker": "Weather160",
    "version": "1.0.0",
    "date": "19/11/25",
    "hour": "15:15",
    "outTemp": "15.8",
    "hiTemp": "15.9",
    "lowTemp": "15.8",
    "outHumidity": "72",
    "dewOut": "10.8",
    "windSpeed": "11.3",
    "windDirection": "ENE",
    "windRun": "0.19",
    "hiSpeed": "19.3",
    "hiDir": "ENE",
    "windChill": "15.4",
    "heatIndex": "15.5",
    "THWIndex": "15.1",
    "THSWIndex": "---",
    "pressure": "755.6",
    "rain": "0.00",
    "rainRate": "0.0",
    "solarRad": "679",
    "solarEnergy": "0.97",
    "hiSolarRad": "679",
    "UVIndex": "3.8",
    "UVDose": "0.03",
    "hiUV": "3.8",
    "headDD": "0.002",
    "coolDD": "0.000",
    "inTemp": "18.8",
    "inHumidity": "63",
    "dewIn": "11.6",
    "inHeat": "18.4",
    "inEMC": "11.63",
    "inAirDensity": "1.1865",
    "2ndTemp": "20.0",
    "2ndHumidity": "50",
    "ET": "0.00",
    "leaf": "0",
    "windSamp": "22",
    "windTx": "1",
    "ISSRecept": "95.7",
    "arcInt": "1",
}

FOCUSER_JSON: Final = {
    "absolute": True,
    "alarm": 0,
    "broker": "Focuser160",
    "cmd": "",
    "connected": True,
    "controller": "Focuser160",
    "device": "2ndMirror",
    "error": "",
    "homing": False,
    "initialized": True,
    "isMoving": False,
    "maxSpeed": 500,
    "maxStep": 50700,
    "position": 32100,
    "tempComp": False,
    "tempCompAvailable": False,
    "temperature": 0,
    "timestamp": "2024-02-27T10:15:48.255",
    "version": "1.0.0",
}

TCS_JSON: Final = {
    "broker": "TCSPD160",
    "version": "20240131",
    "cmd": "",
    "objectName": "NO_OBJ",
    "raAcquis": "02:25:32,1",
    "decAcquis": "+03:22:12,1",
    "epochAcquis": "2000.0",
    "airMass": "1.000",
    "julianDate": "2460368.05207",
    "sideralTime": "20:40:07",
    "hourAngle": "00:00:00",
    "date": "27/02/24",
    "time": "10:14:59",
    "rightAscention": "20 40 07",
    "declination": "-22 35 28",
    "wsDefault": "OFF",
    "guideStr": "27/02/24 20:40:07 20 40 07 -22 35 28",
    "guideAng": "   0.00",
    "guideNor": "  97.00",
    "guideEsp": "S",
    "guideCas": "N",
    "guidePlaca": "0.091",
    "statShutter": "",
    "posCup": "",
    "raOnTarget": True,
    "decOnTarget": True,
    "dome": False,
    "domeOnTarget": True,
    "guider": False,
    "mount": False,
    "grossMovement": False,
    "fineMovement": False,
    "objCentrado": False,
    "varTracking": False,
    "shutter": False,
}
