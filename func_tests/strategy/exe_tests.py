import json
from datetime import datetime, timedelta, timezone
from time import sleep

import func_tests.data_types as data_types

from .test_strategy import Test_Strategy


class E001(Test_Strategy):
    _test_code = "E001"

    def run_test(self) -> None:
        delay = self.wait_2_pub_msgs()
        if delay.seconds < 1:
            self.set_result("error", "Interval between pub msgs smaller than 1 s")
        self._component.send_command("EXPOSE")
        self.wait_acquisition_start()
        delay = self.wait_2_pub_msgs()
        self.wait_return_to_idle()
        if delay.seconds > 0.2:
            self.set_result("error", "Interval between pub msgs larger than 0.2 s")
        return super().run_test()


class E003(Test_Strategy):
    _test_code = "E003"

    def run_test(self) -> None:
        self._component.send_command("_CRITICAL_LOG_")
        self._default_cam_config["INITIAL_LINE"] = 1025
        self._component.set_cam_config(self._default_cam_config)

        with open(self.events_log_file) as file:
            file_content = file.read()
        for level in data_types.Log_Level:
            if self.acs_log_level.value > level.value:
                continue
            if level.name not in file_content:
                self.set_result("error", f"Log level {level.name} not found")

        self._default_cam_config["INITIAL_LINE"] = 1024
        self._component.set_cam_config(self._default_cam_config)
        sleep(2)

        return super().run_test()


class E005(Test_Strategy):
    _test_code = "E005"

    def run_test(self) -> None:
        commands_list = [
            "SET EXPTIME 2",
            "WRITE_SETUP {}",
            "STOP_APP",
            "WAIT_EXPOSE_COMMAND OFF",
        ]
        time_stamp_1 = datetime.now(timezone.utc)
        self._component.send_command("EXPOSE")
        self.wait_acquisition_start()
        for command in commands_list:
            self._component.send_command(command)
        self.wait_acquisition_finish()

        lines_list = self.get_log_file_lines()
        filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp_1)
        filtered_log_lines = self.filter_logs_by_str(filtered_log_lines, "WARNING")
        filtered_log_lines = self.extract_log_msg(filtered_log_lines)

        for command in commands_list:
            command = command.split(" ")[0]
            if f"The {command} command was ignored" not in filtered_log_lines:
                self.set_result("error", f"Log msg related to {command} cmd not found")

        return super().run_test()


class E007(Test_Strategy):
    _test_code = "E007"

    def run_test(self) -> None:
        self._default_cam_config["INITIAL_LINE"] = 1025
        self._component.set_cam_config(self._default_cam_config)
        sleep(2)
        if not self._component.camera.verify_opmode_err():
            self.set_result("error", "The error msg was not found")
        cmd = "EXPOSE"
        self.send_unexpected_command(cmd)

        self._default_cam_config["INITIAL_LINE"] = 1024
        self._component.set_cam_config(self._default_cam_config)
        sleep(2)

        return super().run_test()


class E009(Test_Strategy):
    _test_code = "E009"

    def run_test(self) -> None:
        cmd = "STOP_ACQUISITION"
        self.send_unexpected_command("STOP_ACQUISITION")

        self._default_acq_config["#CYCLES"] = 3
        self._component.set_acquisition_config(self._default_acq_config)
        self.validate_acq_config()
        self._component.send_command("EXPOSE")
        self.wait_acquisition_start()
        self._component.send_command(cmd)
        self.wait_end_of_cycle(1)
        sleep(2)
        if self._component.camera.cam_status.cycles_done != 1:
            self.set_result("error", f"{cmd} command failed")

        return super().run_test()


class E010(Test_Strategy):
    _test_code = "E010"

    def run_test(self) -> None:
        cmd = "PAUSE_ACQUISITION"
        self.send_unexpected_command("PAUSE_ACQUISITION")

        self._default_acq_config["#CYCLES"] = 3
        self._component.set_acquisition_config(self._default_acq_config)
        self.validate_acq_config()

        self._component.send_command("EXPOSE")
        self.wait_acquisition_start()
        self._component.send_command(cmd)
        self.wait_end_of_cycle(1)
        sleep(2)
        if self._component.camera.cam_status.status != "ACQUISITION_PAUSED":
            self.set_result("error", f"{cmd} command failed")
        self._component.send_command("RESUME_ACQUISITION")
        self.wait_acquisition_finish()

        self._default_acq_config["#CYCLES"] = 1
        self._component.set_acquisition_config(self._default_acq_config)

        return super().run_test()


class E011(Test_Strategy):
    _test_code = "E011"

    def run_test(self) -> None:
        self.send_unexpected_command("RESUME_ACQUISITION")
        return super().run_test()


class E012(Test_Strategy):
    _test_code = "E012"

    def run_test(self) -> None:
        cmd = "ABORT_ACQUISITION"
        self.send_unexpected_command("ABORT_ACQUISITION")

        self._default_acq_config["EXPTIME"] = 5
        self._component.set_acquisition_config(self._default_acq_config)
        self.validate_acq_config()
        self._component.send_command("EXPOSE")
        self.wait_acquisition_start()
        self._component.send_command(cmd)
        sleep(2)
        if self._component.camera.cam_status.status != "ACQUISITION_ABORTED":
            self.set_result("error", f"{cmd} command failed")

        self._default_acq_config["EXPTIME"] = 2
        self._component.set_acquisition_config(self._default_acq_config)
        return super().run_test()


class E013(Test_Strategy):  # TODO: este teste precisa ser mudado
    _test_code = "E013"

    def run_test(self) -> None:
        self._default_acq_config["WAVEPLATE_POS"] = 11
        self._component.set_acquisition_config(self._default_acq_config)
        self.validate_acq_config()
        self._component.send_command("EXPOSE")
        self.wait_acquisition_finish()
        sleep(2)
        # if self._component.camera.cam_status.status != "ACQUISITION_ABORTED":
        #     self.set_result("error", f"{cmmd} command failed")
        self._default_acq_config["WAVEPLATE_POS"] = 1
        self._component.set_acquisition_config(self._default_acq_config)

        return super().run_test()


class E019(Test_Strategy):
    _test_code = "E019"

    def run_test(self) -> None:
        time_stamp_1 = datetime.now(timezone.utc)
        self._component.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.wait_acquisition_finish()

        lines_list = self.get_log_file_lines()
        filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp_1)
        filtered_log_lines = self.filter_logs_by_str(filtered_log_lines, "DEBUG")
        filtered_log_lines = self.extract_log_msg(filtered_log_lines)

        expected_strings = [
            "The acquisition of the image series has been finished",
            "The image series has been saved",
        ]
        for _str in expected_strings:
            if _str not in filtered_log_lines:
                self.set_result("error", "Expected log msg was not found")
        return super().run_test()


class E020(Test_Strategy):
    _test_code = "E020"

    def run_test(self) -> None:
        self._default_acq_config["COOLER_POWER_STATUS"] = 1
        self._component.set_acquisition_config(self._default_acq_config)
        self.validate_acq_config()
        self._component.send_command("STOP_APP")
        sleep(2)
        self._default_acq_config["COOLER_POWER_STATUS"] = 0
        self._component.camera.requested_acq_config = self._default_acq_config
        if not self._component.validate_acq_config():
            self.set_result("error", "Unexpected acquisition configuration")

        return super().run_test()
