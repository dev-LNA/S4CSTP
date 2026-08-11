import configparser
import logging
from enum import Enum, IntEnum, StrEnum, auto
from ipaddress import IPv4Address, IPv6Address
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

import func_tests.utils as utils

_log_levels = {
    "0": "STATUS",
    "1": logging.DEBUG,
    "2": logging.INFO,
    "3": logging.WARNING,
    "4": logging.ERROR,
    "5": logging.CRITICAL,
}


class Camera_Configuration(BaseModel):
    em_mode: int
    em_gain: int
    frame_transfer: bool
    readout_rate: float
    preamp: int
    read_mode: int
    acquisition_mode: int
    trigger_mode: int
    vertical_clock_voltage: int
    vertical_shift_speed: int
    shutter_mode: int
    shutter_ttl: int
    shutter_opening_time: int
    shutter_closing_time: int
    initial_column: int
    initial_line: int
    final_column: int
    final_line: int
    vbin: int
    hbin: int
    ad_channel: int

    @classmethod
    def from_dict(cls, cam_config: dict) -> Camera_Configuration:
        em_mode = cam_config["EM_MODE"]
        em_gain = cam_config["EM_GAIN"]
        frame_transfer = cam_config["FRAME_TRANSFER"]
        readout_rate = cam_config["READOUT_RATE"]
        preamp = cam_config["PREAMP"]
        read_mode = cam_config["READ_MODE"]
        acquisition_mode = cam_config["ACQUISITION_MODE"]
        trigger_mode = cam_config["TRIGGER_MODE"]
        vertical_clock_voltage = cam_config["VERTICAL_CLOCK_VOLTAGE"]
        vertical_shift_speed = cam_config["VERTICAL_SHIFT_SPEED"]
        shutter_mode = cam_config["SHUTTER_MODE"]
        shutter_ttl = cam_config["SHUTTER_TTL"]
        shutter_opening_time = cam_config["SHUTTER_OPENING_TIME"]
        shutter_closing_time = cam_config["SHUTTER_CLOSING_TIME"]
        initial_column = cam_config["INITIAL_COLUMN"]
        initial_line = cam_config["INITIAL_LINE"]
        final_column = cam_config["FINAL_COLUMN"]
        final_line = cam_config["FINAL_LINE"]
        vbin = cam_config["VBIN"]
        hbin = cam_config["HBIN"]
        ad_channel = cam_config["AD_CHANNEL"]
        return Camera_Configuration(
            em_mode=em_mode,
            em_gain=em_gain,
            frame_transfer=frame_transfer,
            readout_rate=readout_rate,
            preamp=preamp,
            read_mode=read_mode,
            acquisition_mode=acquisition_mode,
            trigger_mode=trigger_mode,
            vertical_clock_voltage=vertical_clock_voltage,
            vertical_shift_speed=vertical_shift_speed,
            shutter_mode=shutter_mode,
            shutter_ttl=shutter_ttl,
            shutter_opening_time=shutter_opening_time,
            shutter_closing_time=shutter_closing_time,
            initial_column=initial_column,
            initial_line=initial_line,
            final_column=final_column,
            final_line=final_line,
            vbin=vbin,
            hbin=hbin,
            ad_channel=ad_channel,
        )


class Acquisition_Configuration(BaseModel):
    exptime: float
    frames: int
    cycles: int
    suffix: str
    cooler: int
    temp: float
    waveplate_pos: int

    @classmethod
    def from_dict(cls, acq_config: dict) -> Acquisition_Configuration:
        exptime = acq_config["EXPTIME"]
        frames = acq_config["#FRAMES"]
        cycles = acq_config["#CYCLES"]
        suffix = acq_config["suffix"]
        cooler = acq_config["COOLER_POWER_STATUS"]
        temp = acq_config["TEMP"]
        waveplate_pos = acq_config["WAVEPLATE_POS"]
        return Acquisition_Configuration(
            exptime=exptime,
            frames=frames,
            cycles=cycles,
            suffix=suffix,
            cooler=cooler,
            temp=temp,
            waveplate_pos=waveplate_pos,
        )


class Camera_Status(BaseModel):
    cycles_done: int
    frames_done: int
    used_disk_space: int
    current_exp_time: float
    acquiring: bool
    last_image_name: str
    status: str
    current_temp: float
    temp_status: str
    serial_number: int
    power: bool
    acs_mode: bool

    @classmethod
    def from_dict(cls, cam_status: dict) -> Camera_Status:
        cycles_done = cam_status["CYCLES_DONE"]
        last_image_name = cam_status["LAST_IMAGE_NAME"]
        used_disk_space = cam_status["USED_DISK_SPACE"]
        frames_done = cam_status["FRAMES_DONE"]
        status = cam_status["CCD_STATUS"]
        current_temp = cam_status["CURRENT_TEMPERATURE"]
        temp_status = cam_status["TEMPERATURE_STATUS"]
        current_exp_time = cam_status["FRAME_EXPOSURE_TIME"]
        serial_number = cam_status["SERIAL_NUMBER"]
        acquiring = cam_status["ACQUIRING"]
        power = cam_status["POWER"]
        acs_mode = cam_status["ACS MODE"]
        return Camera_Status(
            cycles_done=cycles_done,
            last_image_name=last_image_name,
            used_disk_space=used_disk_space,
            frames_done=frames_done,
            status=status,
            current_temp=current_temp,
            temp_status=temp_status,
            current_exp_time=current_exp_time,
            serial_number=serial_number,
            acquiring=acquiring,
            power=power,
            acs_mode=acs_mode,
        )


class Communication_Status(BaseModel):
    s4gui: bool
    s4ics: bool
    tcs: bool
    weather_st: bool
    focuser: bool

    @classmethod
    def from_dict(cls, comm_status: dict) -> Communication_Status:
        return Communication_Status(
            s4gui=comm_status["S4GUI"],
            s4ics=comm_status["S4ICS"],
            tcs=comm_status["TCS"],
            weather_st=comm_status["Weather Station"],
            focuser=comm_status["Tel. focuser"],
        )


class Error_Type(BaseModel):
    status: bool
    code: int
    source: str

    @classmethod
    def from_dict(cls, error: dict) -> Error_Type:
        return Error_Type(
            status=error["status"], code=error["code"], source=error["source"]
        )


class Execution_Status(Enum):
    NONE = auto()
    IDLE = auto()
    BUSY = auto()
    COMPLETED = auto()
    ERROR = auto()
    TIMEOUT = auto()


class Led_Status(StrEnum):
    OFF = "off"
    ON = "on"
    ERROR = "error"
    WARNING = "warning"


class Command:
    def __init__(self, command_str: str) -> None:
        self._str: str = command_str
        self.__dict: dict = {}
        self.command_len: int = 0
        self._valid: bool = False
        self._supported: str = "off"
        self._timeout: str = "off"
        self._executed: str = "off"

    @property
    def str(self) -> str:
        return self._str

    @property
    def valid(self) -> bool:
        return self._valid

    @property
    def _dict(self) -> dict:
        return self.__dict

    @property
    def supported(self) -> str:
        return self._supported

    @supported.setter
    def supported(self, val: str) -> None:
        self._supported = val

    @property
    def timeout(self) -> str:
        return self._timeout

    @timeout.setter
    def timeout(self, val: str) -> None:
        self._timeout = val

    @property
    def executed(self) -> str:
        return self._executed

    @executed.setter
    def executed(self, val: str) -> None:
        self._executed = val

    def validate(self) -> None:
        splitted_command = self._str.split(" ")
        self.command_len = len(splitted_command)
        self._valid = self.command_len <= 3
        for idx, word in enumerate(splitted_command):
            self.__dict[f"field{idx + 1}"] = word


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
        parser = utils.read_config_file(file_name)
        section_name = "channel configuration"
        return S4ACS_Config(
            channel=int(parser.get(section_name, "channel")),
            acs_mode=int(parser.get(section_name, "ACS mode")) == 1,
            image_path=Path(parser.get(section_name, "image path")),
            log_file_path=Path(parser.get(section_name, "log file path")),
            log_level=Log_Level(_log_levels[parser.get(section_name, "log level")]),
        )


class End_Point(BaseModel):
    ip: str
    port: int = Field(ge=0, le=65535)

    @field_validator("ip")
    def validate_ip(cls, ip) -> IPv4Address | IPv6Address:
        from ipaddress import ip_address

        ip_address(ip)
        return ip

    def to_str(self) -> str:
        return f"tcp://{self.ip}:{self.port}"

    @classmethod
    def from_str(cls, end_point: str) -> End_Point:
        ip, port = end_point.split(":")
        return End_Point(ip=ip, port=int(port))


class Log_Level(IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Test_Result(BaseModel):
    success: str
    test_code: str
    message: str
