import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from time import sleep

import func_tests.component as component
import func_tests.data_types as data_types
import func_tests.utils as utils


class Test_Strategy(ABC):
    _test_code = "A000"

    _min_iteration_time = 0.05  # s
    _timeout_time = 1

    def __init__(self) -> None:
        logging.info(f"Running test {self._test_code}...")
        self._s4acs: component.S4ACS
        self._commands_list: list[data_types.Command]
        self.result = data_types.Test_Result(success="off", test_code="", message="")
        self._create_today_str()
        self.cfg_file_content = utils.read_config_file()
        self.events_log_file = self.cfg_file_content.log_file_path / (
            self._today_str + "_events.log"
        )

        self._default_cam_config = utils.default_cam_config.copy()
        self._default_acq_config = utils.default_acq_config.copy()

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

    @abstractmethod
    def run_test(self) -> None:
        self.set_result("on", "Done")
        logging.debug(f"Result: {self.result.model_dump()}")

    def set_s4acs(self, component: component.S4ACS) -> None:
        self._s4acs = component

    def send_unexpected_command(self, cmd: str) -> None:
        time_stamp_1 = datetime.now(timezone.utc)
        self._s4acs.send_command(cmd)
        sleep(self._timeout_time)

        lines_list = self.get_log_file_lines()
        filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp_1)
        filtered_log_lines = self.filter_logs_by_str(filtered_log_lines, "WARNING")
        filtered_log_lines = self.extract_log_msg(filtered_log_lines)

        if f"The {cmd} command was ignored" != filtered_log_lines[0]:
            self.set_result("error", f"Log msg related to {cmd} cmd not found")
        return

    def validate_acq_config(self) -> None:
        sleep(self._timeout_time)
        if not self._s4acs.validate_acq_config():
            self.set_result("error", "Unexpected acquisition configuration.")
        return

    def calculate_pub_delay(self) -> timedelta:
        while not self._s4acs._subscriber.new_msg:
            sleep(self._min_iteration_time)
        time_stamp_1 = datetime.now()
        while self._s4acs._subscriber.new_msg:
            sleep(self._min_iteration_time)
        while not self._s4acs._subscriber.new_msg:
            sleep(self._min_iteration_time)
        time_stamp_2 = datetime.now()
        return time_stamp_2 - time_stamp_1

    # ========================== WAIT FUNCTIONS ==========================

    def wait_acquisition_start(self) -> None:
        while self._s4acs.camera.cam_status.status != "ACTIVE":
            sleep(self._min_iteration_time)
        logging.debug("The acquisition has started")
        return

    def wait_return_to_idle(self) -> None:
        while self._s4acs.exe_status != "IDLE":
            sleep(self._min_iteration_time)
        logging.debug("S4ACS is in IDLE state")
        return

    def wait_acquisition_finish(self) -> None:
        while (
            self._s4acs.camera.cam_status.cycles_done
            != self._s4acs.camera.requested_acq_config.cycles
        ):
            sleep(self._min_iteration_time)
        logging.debug("Acquisition has been finished")

    def wait_end_of_cycle(self, cycle: int) -> None:
        while self._s4acs.camera.cam_status.cycles_done != cycle:
            sleep(self._min_iteration_time)
        logging.debug(f"This is the end of cycle {cycle}")
        return

    def wait_comm(self, condition: bool) -> bool:
        for _ in range(80):
            if self._s4acs.return_comm_status() is condition:
                logging.debug("Communication condition was reached")
                return True
            sleep(self._min_iteration_time)
        return False

    def wait_cam_on(self) -> None:
        while not self._s4acs.camera.cam_status.power:
            sleep(self._min_iteration_time)
        logging.debug("The cameras was initialized")
        return

    # ========================== LOG FILE =============================

    def get_log_file_lines(self) -> list[str]:
        with open(self.events_log_file) as file:
            lines_list = file.read().splitlines()
        return [line for line in lines_list if "-->" in line]

    def filter_logs_by_timestamp(
        self, lines_list: list[str], time_stamp_1: datetime
    ) -> list[str]:
        filtered_lines = []
        for line in lines_list:
            time_stamp_str = line.split(" ")[0] + " +0000"
            time_stamp_2 = datetime.strptime(time_stamp_str, "%Y-%m-%dT%H:%M:%S.%f %z")
            if time_stamp_2 > time_stamp_1:
                filtered_lines.append(line)
        return filtered_lines

    def filter_logs_by_str(self, lines_list: list[str], _str: str) -> list[str]:
        filtered_lines = []
        for line in lines_list:
            if _str in line:
                filtered_lines.append(line)
        return filtered_lines

    def extract_log_msg(self, lines_list: list[str]) -> list[str]:
        if len(lines_list) == 0:
            return [""]
        return [line.split("--> ")[1] for line in lines_list]

    # ========================== CFG FILE =============================


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
