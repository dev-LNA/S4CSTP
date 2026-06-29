from datetime import datetime, timezone
from time import sleep

import func_tests.data_types as data_types
import func_tests.utils as utils

from .test_strategy import Test_Strategy


class E001(Test_Strategy):
    _test_code = "E001"

    def run_test(self) -> None:
        for _ in range(20):
            delay = self.calculate_pub_delay()
            if delay.total_seconds() > 1:
                break
        else:
            self.set_result("error", "Interval between pub msgs smaller than 1 s")
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        for _ in range(20):
            delay = self.calculate_pub_delay()
            if delay.microseconds > 0.2:
                break
        else:
            self.set_result("error", "Interval between pub msgs smaller than 0.2 s")
        self.wait_acquisition_finish()
        return super().run_test()


class E002(Test_Strategy):
    _test_code = "E002"

    def run_test(self) -> None:
        time_stamp = datetime.now(timezone.utc)

        for external_app in ["s4gui", "s4ics", "tcs", "focuser", "weather_st"]:
            self.framework._do_not_pub = [external_app]
            if not self.wait_comm_ext_app(external_app, False):
                self.set_result(
                    "error", f"The condition {False} was not met: {external_app}"
                )
            self.framework._do_not_pub = []
            if not self.wait_comm_ext_app(external_app, True):
                self.set_result(
                    "error", f"The condition {True} was not met: {external_app}"
                )

        lines_list = self.get_log_file_lines()
        filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp)
        filtered_log_lines = self.filter_logs_by_str(filtered_log_lines, "WARNING")
        filtered_log_lines = self.extract_log_msg(filtered_log_lines)

        for external_app in ["GUI", "ICS", "TCS", "FOCUSER", "WSTATION"]:
            if (
                f"The communication with {external_app} was lost"
                not in filtered_log_lines
            ):
                self.set_result("error", f"Log msg related to {external_app} not found")

            if (
                f"The communication with {external_app} was reestablished"
                not in filtered_log_lines
            ):
                self.set_result("error", f"Log msg related to {external_app} not found")

        return super().run_test()


class E003(Test_Strategy):
    _test_code = "E003"

    def run_test(self) -> None:
        self.s4acs.send_command("_CRITICAL_LOG_")
        self._default_cam_config["INITIAL_LINE"] = 1025
        self.s4acs.set_cam_config(self._default_cam_config)

        with open(self.events_log_file) as file:
            file_content = file.read()
        for level in data_types.Log_Level:
            if self.cfg_file_content.log_level.value > level.value:
                continue
            if level.name not in file_content:
                self.set_result("error", f"Log level {level.name} not found")

        self._default_cam_config["INITIAL_LINE"] = 1024
        self.s4acs.set_cam_config(self._default_cam_config)
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
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        for command in commands_list:
            self.s4acs.send_command(command)
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


class E006(Test_Strategy):
    _test_code = "E006"

    def run_test(self) -> None:
        self.s4acs.send_command("STOP_APP")
        sleep(1)

        if not self.wait_comm(False):
            self.set_result("error", "S4ACS did not stop")

        utils.run_s4acs_exe()
        if not self.wait_comm(True):
            self.set_result("error", "S4ACS did not initialize")

        return super().run_test()


class E007(Test_Strategy):
    _test_code = "E007"

    def run_test(self) -> None:
        self._default_cam_config["INITIAL_LINE"] = 1025
        self.s4acs.set_cam_config(self._default_cam_config)
        sleep(2)
        if not self.s4acs.camera.verify_opmode_err():
            self.set_result("error", "The error msg was not found")
        cmd = "EXPOSE"
        self.send_unexpected_command(cmd)

        self._default_cam_config["INITIAL_LINE"] = 1024
        self.s4acs.set_cam_config(self._default_cam_config)
        sleep(2)

        return super().run_test()


class E009(Test_Strategy):
    _test_code = "E009"

    def run_test(self) -> None:
        cmd = "STOP_ACQUISITION"
        self.send_unexpected_command("STOP_ACQUISITION")

        self._default_acq_config["#CYCLES"] = 3
        self.s4acs.set_acquisition_config(self._default_acq_config)
        self.validate_acq_config()
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.s4acs.send_command(cmd)
        self.wait_end_of_cycle(1)
        sleep(2)
        if self.s4acs.camera.cam_status.cycles_done != 1:
            self.set_result("error", f"{cmd} command failed")

        return super().run_test()


class E010(Test_Strategy):
    _test_code = "E010"

    def run_test(self) -> None:
        cmd = "PAUSE_ACQUISITION"
        self.send_unexpected_command("PAUSE_ACQUISITION")

        self._default_acq_config["#CYCLES"] = 3
        self.s4acs.set_acquisition_config(self._default_acq_config)
        self.validate_acq_config()

        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.s4acs.send_command(cmd)
        self.wait_end_of_cycle(1)
        sleep(2)
        if self.s4acs.camera.cam_status.status != "ACQUISITION_PAUSED":
            self.set_result("error", f"{cmd} command failed")
        self.s4acs.send_command("RESUME_ACQUISITION")
        self.wait_acquisition_finish()

        self._default_acq_config["#CYCLES"] = 1
        self.s4acs.set_acquisition_config(self._default_acq_config)

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
        self.s4acs.set_acquisition_config(self._default_acq_config)
        self.validate_acq_config()
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.s4acs.send_command(cmd)
        sleep(2)
        if self.s4acs.camera.cam_status.status != "ACQUISITION_ABORTED":
            self.set_result("error", f"{cmd} command failed")

        self._default_acq_config["EXPTIME"] = 2
        self.s4acs.set_acquisition_config(self._default_acq_config)
        return super().run_test()


class E013(Test_Strategy):  # TODO: este teste precisa ser mudado
    _test_code = "E013"

    def run_test(self) -> None:
        self._default_acq_config["WAVEPLATE_POS"] = 11
        self.s4acs.set_acquisition_config(self._default_acq_config)
        self.validate_acq_config()
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_finish()
        sleep(2)
        # if self.s4acs.camera.cam_status.status != "ACQUISITION_ABORTED":
        #     self.set_result("error", f"{cmmd} command failed")
        self._default_acq_config["WAVEPLATE_POS"] = 1
        self.s4acs.set_acquisition_config(self._default_acq_config)

        return super().run_test()


class E019(Test_Strategy):
    _test_code = "E019"

    def run_test(self) -> None:
        time_stamp_1 = datetime.now(timezone.utc)
        self.s4acs.send_command("EXPOSE")
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
