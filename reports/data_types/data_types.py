import logging
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

from astropy.io import fits
from pydantic import BaseModel, Field
from utils import read_config_file

_log_levels = {
    "0": "STATUS",
    "1": logging.DEBUG,
    "2": logging.INFO,
    "3": logging.WARNING,
    "4": logging.ERROR,
    "5": logging.CRITICAL,
}


class Log_Level(IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class S4ACS_Config(BaseModel):
    channel: int
    acs_mode: int
    image_path: Path
    log_file_path: Path
    log_level: Log_Level

    def to_sparc4_format(self) -> dict[str, Any]:
        new_dict = {
            key.replace("_", " "): val for key, val in self.model_dump().items()
        }
        new_dict["log level"] //= 10
        return new_dict

    @classmethod
    def from_config_file(cls, file_name: str) -> S4ACS_Config:
        parser = read_config_file(file_name)
        section_name = "channel configuration"
        return S4ACS_Config(
            channel=int(parser.get(section_name, "channel")),
            acs_mode=int(parser.get(section_name, "ACS mode")) == 1,
            image_path=Path(parser.get(section_name, "image path")),
            log_file_path=Path(parser.get(section_name, "log file path")),
            log_level=Log_Level(_log_levels[parser.get(section_name, "log level")]),
        )


class Header_Content(BaseModel):
    nframes: int = Field(gt=0)
    frameind: int = Field(gt=0)
    nseq: int = Field(gt=0)
    seqindex: int = Field(gt=0)
    ncycles: int = Field(gt=0)
    cyclind: int = Field(gt=0)
    exptime: float
    filename: str
    instmode: Literal["PHOT", "POLAR"]
    date_obs: str
    obstype: Literal["ZERO", "FLAT", "OBJECT", "FOCUS", "DARK"]

    @classmethod
    def from_header(cls, hdr: fits.Header) -> Header_Content:
        _hdr: dict[str, Any] = {k.lower(): v for k, v in hdr.items()}
        _hdr["date_obs"] = _hdr["date-obs"]
        return Header_Content(**_hdr)

    @property
    def to_csv_format(self) -> dict[str, Any]:
        return {
            "FILENAME": self.filename,
            "DATE-OBS": self.date_obs,
            "OBSTYPE": self.obstype,
            "INSTMODE": self.instmode,
            "CYCLE": self.ncycles,
            "CYCLE_IDX": self.cyclind,
            "SEQ": self.nseq,
            "SEQ_IDX": self.seqindex,
            "FRAME": self.nframes,
            "FRAME_IDX": self.frameind,
            "EXPTIME": self.exptime,
        }
