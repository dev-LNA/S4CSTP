import logging
from dataclasses import dataclass
from enum import Enum, Flag, IntEnum, StrEnum, auto
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Container
from unittest.mock import Base

import zmq
from pydantic import BaseModel, Field, field_validator

import src.client as client
import src.comm_protocol as comm_protocol
import src.component.component as component
from src.component import EACS, EICS, Fake_Component, Focuser


class Camera_Configuration(BaseModel):
    read_rate: float
    pre_amp: int
    read_mode: int
    acq_mode: int
    trigger_mode: int
    vclock_voltage: int
    shutter: int
    shutter_ttl: int
    shutter_op_time: int
    shutter_cls_time: int
    init_col: int
    init_line: int
    final_col: int
    final_line: int
    vbin: int
    hbin: int
    ad_chnnl: int

    @classmethod
    def from_dict(cls, cam_config: dict) -> Camera_Configuration:
        read_rate = cam_config["READOUT_RATE"]
        pre_amp = cam_config["PREAMP"]
        read_mode = cam_config["READ_MODE"]
        acq_mode = cam_config["ACQUISITION_MODE"]
        trigger_mode = cam_config["TRIGGER_MODE"]
        vclock_voltage = cam_config["VERTICAL_CLOCK_VOLTAGE"]
        shutter = cam_config["SHUTTER_MODE"]
        shutter_ttl = cam_config["SHUTTER_TTL"]
        shutter_op_time = cam_config["SHUTTER_OPENING_TIME"]
        shutter_cls_time = cam_config["SHUTTER_CLOSING_TIME"]
        init_col = cam_config["INITIAL_COLUMN"]
        init_line = cam_config["INITIAL_LINE"]
        final_col = cam_config["FINAL_COLUMN"]
        final_line = cam_config["FINAL_LINE"]
        vbin = cam_config["VBIN"]
        hbin = cam_config["HBIN"]
        ad_chnnl = cam_config["AD_CHANNEL"]
        return Camera_Configuration(
            read_rate=read_rate,
            pre_amp=pre_amp,
            read_mode=read_mode,
            acq_mode=acq_mode,
            trigger_mode=trigger_mode,
            vclock_voltage=vclock_voltage,
            shutter=shutter,
            shutter_ttl=shutter_ttl,
            shutter_op_time=shutter_op_time,
            shutter_cls_time=shutter_cls_time,
            init_col=init_col,
            init_line=init_line,
            final_col=final_col,
            final_line=final_line,
            vbin=vbin,
            hbin=hbin,
            ad_chnnl=ad_chnnl,
        )


class Acquisition_Configuration(BaseModel):
    exp_time: float
    frames: int
    cycles: int
    suffix: str
    cooler: int
    temperature: float

    @classmethod
    def from_dict(cls, acq_config: dict) -> Acquisition_Configuration:
        exp_time = acq_config["EXPTIME"]
        frames = acq_config["#FRAMES"]
        cycles = acq_config["#CYCLES"]
        suffix = acq_config["suffix"]
        cooler = acq_config["COOLER_POWER_STATUS"]
        temperature = acq_config["TEMP"]
        return Acquisition_Configuration(
            exp_time=exp_time,
            frames=frames,
            cycles=cycles,
            suffix=suffix,
            cooler=cooler,
            temperature=temperature,
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


class Level(Enum):
    NONE = auto()
    VERY_LOW = auto()
    LOW = auto()
    NORMAL = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


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


class Log_Level(IntEnum):  # TODO adicionar ao diagrama de classes
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Light_Sources(Flag):
    NONE = auto()
    DARK = auto()
    SHUTTER = auto()
    THORIUM = auto()
    HALOGEN = auto()
    CALIB = auto()
    OBJECT = auto()
    SKY = auto()


class Request:
    def __init__(self, request_str: str) -> None:
        self._str: str = request_str
        self.__dict: dict = {}
        self.req_len: int = 0
        self._valid: bool = False
        self._status: str = "off"
        self._recipient: str = "off"

    @property
    def str(self) -> str:
        return self._str

    @property
    def valid(self) -> bool:
        return self._valid

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, val: str) -> None:
        self._status = val

    @property
    def recipient(self) -> str:
        return self._recipient

    @recipient.setter
    def recipient(self, val: str) -> None:
        self._recipient = val

    @property
    def _dict(self) -> dict:
        return self.__dict

    @property
    def command(self) -> str:
        return self._str.split(" ", 1)[1]

    def validate(self) -> None:
        splitted_req = self._str.split(" ")
        self.req_len = len(splitted_req)
        self._valid = self.req_len <= 4
        for idx, word in enumerate(splitted_req):
            self.__dict[f"field{idx + 1}"] = word


class Command:  # TODO: add to classes diagram
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


@dataclass
class Components_Container:
    eacs1: component.Component
    eacs2: component.Component
    eics: component.Component
    focuser: component.Component

    @property
    def values(self) -> list[component.Component]:
        return list(self.__dict__.values())

    @property
    def keys(self) -> list:
        return [key.upper() for key in self.__dict__.keys()]

    @property
    def items(self) -> tuple:
        return tuple(self.__dict__.items())

    @property
    def _dict(self) -> dict[str, component.Component]:
        return {key.upper(): val for (key, val) in self.__dict__.items()}


class Mediator_Setup:
    def __init__(self) -> None:
        end_point = End_Point(ip="192.168.0.1", port=5555)
        subscriber = comm_protocol.Fake_Subscriber(end_point)
        requester = comm_protocol.Fake_Requester(end_point)
        eacs1 = Fake_Component(None, subscriber, requester)
        eacs2 = Fake_Component(None, subscriber, requester)
        eics = Fake_Component(None, subscriber, requester)
        focuser = Fake_Component(None, subscriber, requester)
        self._fake_container = Components_Container(
            eacs1=eacs1,
            eacs2=eacs2,
            eics=eics,
            focuser=focuser,
        )
        replier = comm_protocol.Fake_Replier(end_point)
        self._fake_client = client.Client(replier)

    def create(self, _type: str) -> tuple:
        """Create a container of Component classes.

        Args:
            _type (str): container to be retrived.
            Allowed values are 'fake', 'test_acs1', 'test_client', 'test_focs',
            'test_acs1_client', 'test_acs1_acs2_client', 'test_all_without_eics'.

        Raises:
            ValueError: an unexpected value was found.

        Returns:
            Tuple: a tupple containing the client and container instanciated by
            the class.
        """
        if _type == "fake":
            return self._fake_client, self._fake_container

        if _type == "test_acs1":
            context = zmq.Context()
            end_point = End_Point(ip="200.131.64.25", port=5555)
            subscriber = comm_protocol.ZeroMQ_SUB(end_point, context)
            end_point = End_Point(ip="200.131.64.25", port=5556)
            requester = comm_protocol.ZeroMQ_REQ(end_point, context)
            eacs1 = EACS(None, subscriber, requester)

            fake_container = self._fake_container
            fake_container.eacs1 = eacs1
            return self._fake_client, fake_container

        if _type == "test_client":
            context = zmq.Context()
            end_point = End_Point(ip="200.131.64.25", port=5557)
            replier = comm_protocol.ZeroMQ_REP(end_point, context)
            _client = client.Client(replier)

            return _client, self._fake_container

        if _type == "test_foc":
            context = zmq.Context()
            end_point = End_Point(ip="200.131.64.25", port=7001)
            subscriber = comm_protocol.ZeroMQ_SUB(end_point, context)
            end_point = End_Point(ip="200.131.64.25", port=7002)
            requester = comm_protocol.Fake_Requester(end_point)
            focuser = Focuser(None, subscriber, requester)

            fake_container = self._fake_container
            fake_container.focuser = focuser
            return self._fake_client, fake_container

        if _type == "test_acs1_client":
            context = zmq.Context()
            end_point = End_Point(ip="200.131.64.25", port=5555)
            subscriber = comm_protocol.ZeroMQ_SUB(end_point, context)
            end_point = End_Point(ip="200.131.64.25", port=5556)
            requester = comm_protocol.ZeroMQ_REQ(end_point, context)
            eacs1 = EACS(None, subscriber, requester)

            end_point = End_Point(ip="200.131.64.25", port=5553)
            replier = comm_protocol.ZeroMQ_REP(end_point, context)
            _client = client.Client(replier)

            fake_container = self._fake_container
            fake_container.eacs1 = eacs1
            return _client, fake_container

        if _type == "test_acs1_acs2_client":
            context = zmq.Context()
            end_point = End_Point(ip="200.131.64.25", port=5555)
            subscriber = comm_protocol.ZeroMQ_SUB(end_point, context)
            end_point = End_Point(ip="200.131.64.25", port=5556)
            requester = comm_protocol.ZeroMQ_REQ(end_point, context)
            eacs1 = EACS(None, subscriber, requester)

            end_point = End_Point(ip="200.131.64.25", port=5557)
            subscriber = comm_protocol.ZeroMQ_SUB(end_point, context)
            end_point = End_Point(ip="200.131.64.25", port=5558)
            requester = comm_protocol.ZeroMQ_REQ(end_point, context)
            eacs2 = EACS(None, subscriber, requester)

            end_point = End_Point(ip="200.131.64.25", port=5559)
            replier = comm_protocol.ZeroMQ_REP(end_point, context)
            _client = client.Client(replier)

            fake_container = self._fake_container
            fake_container.eacs1 = eacs1
            fake_container.eacs2 = eacs2
            return _client, fake_container

        if _type == "test_all_without_eics":
            context = zmq.Context()
            end_point = End_Point(ip="200.131.64.25", port=5555)
            subscriber = comm_protocol.ZeroMQ_SUB(end_point, context)
            end_point = End_Point(ip="200.131.64.25", port=5556)
            requester = comm_protocol.ZeroMQ_REQ(end_point, context)
            eacs1 = EACS(None, subscriber, requester)

            end_point = End_Point(ip="200.131.64.25", port=5557)
            subscriber = comm_protocol.ZeroMQ_SUB(end_point, context)
            end_point = End_Point(ip="200.131.64.25", port=5558)
            requester = comm_protocol.ZeroMQ_REQ(end_point, context)
            eacs2 = EACS(None, subscriber, requester)

            end_point = End_Point(ip="200.131.64.25", port=7001)
            subscriber = comm_protocol.ZeroMQ_SUB(end_point, context)
            end_point = End_Point(ip="200.131.64.25", port=7002)
            requester = comm_protocol.Fake_Requester(end_point)
            focuser = Focuser(None, subscriber, requester)

            end_point = End_Point(ip="200.131.64.25", port=5553)
            replier = comm_protocol.ZeroMQ_REP(end_point, context)
            _client = client.Client(replier)

            fake_container = self._fake_container
            fake_container.eacs1 = eacs1
            fake_container.eacs2 = eacs2
            fake_container.focuser = focuser
            return _client, fake_container

        else:
            raise ValueError(f"The provided type {_type} was not found.")
