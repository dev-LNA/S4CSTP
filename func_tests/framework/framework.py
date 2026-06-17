import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import func_tests.component as component
import func_tests.data_types as data_types
import func_tests.gui as gui
import func_tests.strategy as strategy
import func_tests.utils as utils


class Functionalities_Tests_Framework:
    stop_thread = False
    start_tests = False
    stop_tests = False
    stop_1st_err: bool

    def __init__(
        self,
        s4acs: component.Component,
        tests_list: Sequence[strategy.Test_Strategy],
        _gui: gui.GUI,
    ) -> None:
        self.s4acs = s4acs
        self.tests_list = tests_list
        self._gui = _gui
        self.log_dir = Path("func_tests/_logs")
        self.log_level = data_types.Log_Level.INFO
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return

    def initialize(self) -> None:
        self.create_log_file()
        logging.info("Framework was started")
        self.s4acs.initialize()
        logging.debug("Setting default configuration...")
        self.s4acs.set_acquisition_config(utils.default_acq_config.copy())
        self.s4acs.set_cam_config(utils.default_cam_config.copy())
        for _test in self.tests_list:
            _test.set_component(self.s4acs)
        logging.debug("Framework was initialized succesfully")
        return

    def create_log_file(self) -> None:
        now = datetime.now(timezone.utc)
        log_file = self.log_dir / f"{now.strftime('%Y%m%d')}.log"
        datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
        self._create_log_file_header(log_file, datetime_str)
        logging.basicConfig(
            filename=log_file,
            level=self.log_level,
            format="%(asctime)s [%(levelname)-8s] --> %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            filemode="a",
            force=True,
        )

    def _create_log_file_header(self, log_file: Path, datetime_str: str) -> None:
        with open(log_file, "w") as file:
            file.write(
                "\n========================================================================================\n"
                "S4ACS Functionalities Tests Framework - Event Log\n"
                "========================================================================================\n\n"
                "Description     : this file logs important events occured during the framework execution\n"
                "Version         : v0.1.0\n"
                "Log Type        : Operational Events\n"
                f"Log level       : {self.log_level.name}\n"
                f"Created at (UTC): {datetime_str}\n"
                "----------------------------------------------------------------------------------------\n"
                "Timestamp           Level          Message\n"
                "----------------------------------------------------------------------------------------\n\n"
            )
        return

    def end(self) -> None:
        self.s4acs.end()
        logging.info("Framework was stopped")

    def get_status(self) -> None:
        while not self.stop_thread:
            self.s4acs.get_status_message()

    def run(self) -> None:
        while not self.stop_thread:
            sleep(0.025)
            if self.start_tests is True:
                self.clear_results()
                self.run_tests()
                logging.info("The tests were finished")
                self._gui.gui_widgets.framework_run_tests_btn.setEnabled(True)
                self._gui.gui_widgets.framework_stop_tests_btn.setDisabled(True)
                self.start_tests = False

    def run_tests(self) -> None:
        logging.info("Starting the tests...")
        for _test in self.tests_list:
            _test._component.send_command(
                f"====== This is the test {_test._test_code} ======"
            )
            _test.set_result("warn", "")
            _test.run_test()

            if self.stop_1st_err and (_test.result.success == "error"):
                logging.info("An error was found. Stopping the tests...")
                return
            if self.stop_tests:
                logging.info("The tests were stopped")
                self.stop_tests = False
                return

    def clear_results(self) -> None:
        for _test in self.tests_list:
            _test.result = data_types.Test_Result(
                success="off", test_code="", message=""
            )
