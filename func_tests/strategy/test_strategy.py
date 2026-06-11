import configparser
import getpass
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import func_tests.component as component
import func_tests.data_types as data_types


class Test_Strategy(ABC):
    _test_code = "A000"
    _log_levels = {
        "0": "STATUS",
        "1": "DEBUG",
        "2": "INFO",
        "3": "WARNING",
        "4": "ERROR",
        "5": "CRITICAL",
    }

    def __init__(self) -> None:
        self._component: component.Component
        self._commands_list: list[data_types.Command]
        self.result = data_types.Test_Result(success="off", test_code="", message="")
        self._today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        self._read_config_file()

    def set_result(self, succes: str, msg: str) -> None:
        if self.result.success == "error":
            return
        self.result = data_types.Test_Result(
            success=succes, test_code=self._test_code, message=msg
        )

    def _read_config_file(self) -> None:
        config_file_folder = Path(f"C:/Users/{getpass.getuser()}/SPARC4/ACS")
        config_file = config_file_folder / "acs_config.cfg"
        if not config_file.exists():
            raise RuntimeError(f"file {config_file} not found")
        config = configparser.ConfigParser()
        config.read(config_file)
        self.channel = config.get("channel configuration", "channel")
        self.acs_mode = config.get("channel configuration", "s4acs mode") == 1
        self.acs_log_level = self._log_levels[
            config.get("channel configuration", "log level")
        ]
        log_folder = config.get("channel configuration", "log file path")
        self.log_folder = Path(log_folder.replace('"', ""))
        imgs_folder = config.get("channel configuration", "image path")
        self.imgs_folder = Path(imgs_folder.replace('"', ""))

    @abstractmethod
    def run_test(self) -> None:
        if self.result is not None:
            logging.debug(f"Result: {self.result.model_dump()}")

    def set_component(self, component: component.Component) -> None:
        self._component = component

    def wait_acquisition_finish(self) -> None:
        while self._component.exe_status != "BUSY":
            self._component.get_status_message()
            sleep(0.3)
        while self._component.exe_status == "BUSY":
            self._component.get_status_message()
            sleep(0.3)
            continue


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
