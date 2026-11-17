from configparser import ConfigParser
from pathlib import Path

import pandas as pd


def read_config_file(file_name: str) -> ConfigParser:
    cfg_file_folder = Path(Path.cwd().root) / "sparc4" / "config"
    cfg_file = cfg_file_folder / file_name
    if not cfg_file.exists():
        raise RuntimeError(f"file {cfg_file} not found")
    config = ConfigParser()
    config.read(cfg_file)

    return config


def create_csv(csv_file: Path) -> None:
    headers = [
        "FILENAME",
        "DATE-OBS",
        "OBSTYPE",
        "INSTMODE",
        "CYCLE",
        "CYCLE_IDX",
        "SEQ",
        "SEQ_IDX",
        "FRAME",
        "FRAME_IDX",
        "EXPTIME",
    ]
    _dict = dict.fromkeys(headers, [])
    df = pd.DataFrame.from_dict(_dict)
    df.to_csv(csv_file, index=False)
