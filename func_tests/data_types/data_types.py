import logging
from collections.abc import Sequence
from enum import Enum, IntEnum, StrEnum, auto
from ipaddress import IPv4Address, IPv6Address

import zmq
from pydantic import BaseModel, Field, field_validator

import func_tests.comm_channel as comm_channel
import func_tests.component as component
import func_tests.strategy as strategy


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


class Component_Creator:
    def create(self, _type: str) -> component.Component:
        """create a Component

        Args:
            _type (str): _description_

        Returns:
            component.Component: _description_
        """
        if _type == "fake":
            end_point = End_Point(ip="192.168.0.1", port=5555)
            subscriber = comm_channel.Fake_Subscriber(end_point)
            requester = comm_channel.Fake_Requester(end_point)
            return component.Fake_Component(subscriber, requester)

        if _type == "real":
            context = zmq.Context()
            end_point = End_Point(ip="200.131.64.25", port=5555)
            subscriber = comm_channel.ZeroMQ_SUB(end_point, context)
            end_point = End_Point(ip="200.131.64.25", port=5556)
            requester = comm_channel.ZeroMQ_REQ(end_point, context)
            return component.Component(subscriber, requester)

        else:
            raise ValueError(f"Unknown type: {_type}")


class Tests_List_Creator:
    def create(self, _type: str) -> Sequence[strategy.Test_Strategy]:
        if _type == "fake":
            return [strategy.Fake_Positive_Test() for _ in range(17)] + [
                strategy.Fake_Negative_Test() for _ in range(10)
            ]

        if _type == "real":
            return [
                strategy.I001(),
                strategy.I002(),
                strategy.I003(),
                strategy.I006(),
                # strategy.E001(),
                strategy.E003(),
                strategy.E005(),
                strategy.E007(),
                strategy.E009(),
                strategy.E010(),
                strategy.E011(),
                strategy.E012(),
                strategy.E013(),
                strategy.E019(),
                strategy.E020(),
            ]
        if _type == "one test":
            return [strategy.I001()]
        else:
            raise ValueError(f"Unknown type: {_type}")
