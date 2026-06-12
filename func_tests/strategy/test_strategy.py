import configparser
import getpass
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

import func_tests.component as component
import func_tests.data_types as data_types


class Test_Strategy(ABC):
    _test_code = "A000"
    _log_levels = {
        "0": "STATUS",
        "1": logging.DEBUG,
        "2": logging.INFO,
        "3": logging.WARNING,
        "4": logging.ERROR,
        "5": logging.CRITICAL,
    }
    _default_cam_config = {
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

    def __init__(self) -> None:
        logging.info(f"Running test {self._test_code}...")
        self._component: component.Component
        self._commands_list: list[data_types.Command]
        self.result = data_types.Test_Result(success="off", test_code="", message="")
        self._create_today_str()
        self._read_config_file()
        self.events_log_file = self.log_folder / (self._today_str + "_events.log")

    def set_result(self, succes: str, msg: str) -> None:
        if self.result.success == "error":
            return
        self.result = data_types.Test_Result(
            success=succes, test_code=self._test_code, message=msg
        )

    def _create_today_str(self) -> None:
        now = datetime.now(timezone.utc)
        if now.hour < 12:
            now -= timedelta(1)
        self._today_str = now.strftime("%Y%m%d")

    def _read_config_file(self) -> None:
        section_name = "channel configuration"
        cfg_file_folder = Path(f"C:/Users/{getpass.getuser()}/SPARC4/ACS")
        cfg_file = cfg_file_folder / "acs_config.cfg"
        if not cfg_file.exists():
            raise RuntimeError(f"file {cfg_file} not found")
        config = configparser.ConfigParser()
        config.read(cfg_file)
        self.channel = config.get(section_name, "channel")
        self.acs_mode = config.get(section_name, "s4acs mode") == 1
        self.acs_log_level = data_types.Log_Level(
            self._log_levels[config.get(section_name, "log level")]
        )
        log_folder = config.get(section_name, "log file path")
        self.log_folder = Path(log_folder.replace('"', ""))
        imgs_folder = config.get(section_name, "image path")
        self.imgs_folder = Path(imgs_folder.replace('"', ""))

    @abstractmethod
    def run_test(self) -> None:
        if self.result is not None:
            logging.debug(f"Result: {self.result.model_dump()}")

    def set_component(self, component: component.Component) -> None:
        self._component = component

    def wait_acquisition_start(self) -> None:
        while self._component.exe_status != "BUSY":
            self._component.get_status_message()
            sleep(0.1)

    def wait_return_to_idle(self) -> None:
        while self._component.exe_status != "IDLE":
            self._component.get_status_message()
            sleep(0.1)

    def wait_acquisition_finish(self) -> None:
        while self._component.exe_status != "BUSY":
            self._component.get_status_message()
            sleep(0.3)
        while self._component.exe_status == "BUSY":
            self._component.get_status_message()
            sleep(0.3)
            continue

    def wait_2_pub_msgs(self) -> timedelta:
        while not self._component._subscriber.new_msg:
            self._component.get_status_message()
        time_stamp_1 = self._component._subscriber.last_msg_timestamp
        while self._component._subscriber.new_msg:
            self._component.get_status_message()
        while not self._component._subscriber.new_msg:
            self._component.get_status_message()
        time_stamp_2 = self._component._subscriber.last_msg_timestamp
        return time_stamp_2 - time_stamp_1


class Fake_Positive_Test(Test_Strategy):
    _test_code = "P000"

    def run_test(self) -> None:
        sleep(0.5)
        self.set_result("on", "Done")
        return super().run_test()


class Fake_Negative_Test(Test_Strategy):
    _test_code = "N000"

    def run_test(self) -> None:
        sleep(0.5)
        self.set_result("error", "Done")

        return super().run_test()
