import json
from datetime import datetime, timezone
from time import sleep

from astropy.io import fits

import func_tests.data_types as data_types
import func_tests.utils as utils
from func_tests.utils.utils import S4GUI_JSON

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
        cam_config = self._default_cam_config.copy()
        cam_config["INITIAL_LINE"] = 1025
        self.s4acs.set_cam_config(cam_config)

        with open(self.events_log_file) as file:
            file_content = file.read()
        for level in data_types.Log_Level:
            if self.acs_config.log_level.value > level.value:
                continue
            if level.name not in file_content:
                self.set_result("error", f"Log level {level.name} not found")

        cam_config["INITIAL_LINE"] = 1024
        self.s4acs.set_cam_config(cam_config)
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
        cam_config = self._default_cam_config.copy()
        cam_config["FINAL_LINE"] = 1025
        self.s4acs.set_cam_config(cam_config)
        sleep(2)
        if not self.s4acs.camera.opmode_err.status:
            self.set_result("error", "The error msg was not found")
        cmd = "EXPOSE"
        self.send_unexpected_command(cmd)

        cam_config["FINAL_LINE"] = 1024
        self.s4acs.set_cam_config(cam_config)
        sleep(2)

        return super().run_test()


class E008(Test_Strategy):
    _test_code = "E008"

    def run_test(self) -> None:
        time_stamp_1 = datetime.now(timezone.utc)
        mechanism_status = utils.S4ICS_MECHANISM_STATUS.copy()
        mechanism_status["condition"] = "BUSY"
        mechanism = utils.S4ICS_MECHANISM.copy()
        mechanism["status"] = mechanism_status
        second_part_s4ics_pub = utils.SECOND_PART_S4ICS_PUB.copy()
        second_part_s4ics_pub["mechanisms"] = [mechanism]

        self.framework._external_apps["s4ics"].status = (
            utils.FIRST_PART_S4ICS_PUB + json.dumps(second_part_s4ics_pub)
        )
        s4gui_json = utils.S4GUI_JSON.copy()
        s4gui_json["INSTMODE"] = "POLAR"
        self.framework._external_apps["s4gui"].status = s4gui_json
        self.s4acs.send_command("WAIT_EXPOSE_COMMAND ON")
        sleep(1)
        self.s4acs.send_command("EXPOSE")
        sleep(2)
        lines_list = self.get_log_file_lines()
        filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp_1)
        debug_logs = self.filter_logs_by_str(filtered_log_lines, "DEBUG")
        debug_logs = self.extract_log_msg(debug_logs)
        if "Verifying the waveplate status" not in debug_logs:
            self.set_result("error", "Log msg related to waveplate not found")

        error_logs = self.filter_logs_by_str(filtered_log_lines, "ERROR")
        error_logs = self.extract_log_msg(error_logs)
        if "The communication with S4ICS has failed" not in error_logs:
            self.set_result("error", "Log msg related to waveplate not found")

        self.framework._external_apps["s4ics"].status = (
            utils.FIRST_PART_S4ICS_PUB + json.dumps(utils.SECOND_PART_S4ICS_PUB)
        )
        self.framework._external_apps["s4gui"].status = utils.S4GUI_JSON.copy()
        self.s4acs.send_command("WAIT_EXPOSE_COMMAND OFF")

        return super().run_test()


class E009(Test_Strategy):
    _test_code = "E009"

    def run_test(self) -> None:
        cmd = "STOP_ACQUISITION"
        time_stamp = datetime.now(timezone.utc)
        self.send_unexpected_command("STOP_ACQUISITION")

        self._default_acq_config["#CYCLES"] = 3
        self.s4acs.set_acquisition_config(self._default_acq_config)
        if not self.s4acs.validate_acq_config():
            self.set_result("error", "Unexpected acquisition configuration.")

        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.s4acs.send_command(cmd)
        self.wait_end_of_cycle(1)
        sleep(2)
        if self.s4acs.camera.cam_status.cycles_done != 1:
            self.set_result("error", f"{cmd} command failed")

        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.wait_end_of_cycle(2)
        self.s4acs.send_command(cmd)

        lines_list = self.get_log_file_lines()
        filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp)
        filtered_log_lines = self.filter_logs_by_str(filtered_log_lines, "WARNING")
        filtered_log_lines = self.extract_log_msg(filtered_log_lines)

        for line in filtered_log_lines:
            if line != f"The {cmd} command was ignored":
                self.set_result("error", f"Unexpected log message: {line}")

        self.wait_end_of_cycle(3)
        self.s4acs.set_acquisition_config(self._default_acq_config)

        return super().run_test()


class E010(Test_Strategy):
    _test_code = "E010"

    def run_test(self) -> None:
        time_stamp = datetime.now(timezone.utc)
        cmd = "PAUSE_ACQUISITION"
        self.send_unexpected_command("PAUSE_ACQUISITION")

        acq_config = self._default_acq_config.copy()
        acq_config["#CYCLES"] = 3
        self.s4acs.set_acquisition_config(acq_config)
        if not self.s4acs.validate_acq_config():
            self.set_result("error", "Unexpected acquisition configuration.")

        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.s4acs.send_command(cmd)
        self.wait_end_of_cycle(1)
        sleep(2)
        if self.s4acs.camera.cam_status.status != "ACQUISITION_PAUSED":
            self.set_result("error", f"{cmd} command failed")
        self.s4acs.send_command("RESUME_ACQUISITION")
        self.wait_acquisition_finish()

        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.wait_end_of_cycle(2)
        self.s4acs.send_command(cmd)

        lines_list = self.get_log_file_lines()
        filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp)
        filtered_log_lines = self.filter_logs_by_str(filtered_log_lines, "WARNING")
        filtered_log_lines = self.extract_log_msg(filtered_log_lines)

        for line in filtered_log_lines:
            if line != f"The {cmd} command was ignored":
                self.set_result("error", f"Unexpected log message: {line}")

        self.wait_end_of_cycle(3)
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
        if not self.s4acs.validate_acq_config():
            self.set_result("error", "Unexpected acquisition configuration.")
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.s4acs.send_command(cmd)
        sleep(2)
        if self.s4acs.camera.cam_status.status != "IDLE":
            self.set_result("error", f"{cmd} command failed")

        self._default_acq_config["EXPTIME"] = 2
        self.s4acs.set_acquisition_config(self._default_acq_config)
        sleep(1)
        return super().run_test()


class E013(Test_Strategy):
    _test_code = "E013"

    def run_test(self) -> None:
        parameters = [
            "PREAMP",
            "EM_MODE",
            "EM_GAIN",
            "INITIAL_LINE",
            "INITIAL_COLUMN",
            "FINAL_LINE",
            "FINAL_COLUMN",
            "VBIN",
            "HBIN",
            "SHUTTER_TTL",
            "SHUTTER_MODE",
            "AD_CHANNEL",
        ]

        for key in parameters:
            self.s4acs.send_command(f"==== Evaluating {key} parameter ====")
            time_stamp = datetime.now(timezone.utc)
            _min, _max = self.s4acs.camera.opmode_params_limits[key]
            self._send_commands_sequence(key, _max, _max + 1)
            self._send_commands_sequence(key, _min, _min - 1)

            lines_list = self.get_log_file_lines()
            filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp)
            filtered_log_lines = self.filter_logs_by_str(filtered_log_lines, "ERROR")
            filtered_log_lines = self.extract_log_msg(filtered_log_lines)
            expected_string = f"The value {_max + 1:.2f} was received for the {key} parameter. However, it should be in the [{_min:.2f}, {_max:.2f}] range."
            self._search_expected_log_msg(filtered_log_lines, expected_string)
            expected_string = f"The value {_min - 1:.2f} was received for the {key} parameter. However, it should be in the [{_min:.2f}, {_max:.2f}] range."
            self._search_expected_log_msg(filtered_log_lines, expected_string)
            self.s4acs.set_cam_config(self._default_cam_config)

        return super().run_test()

    def _send_commands_sequence(self, key: str, val: int, adjusted_val: int) -> None:
        cam_config = self._default_cam_config.copy()
        cam_config[key] = adjusted_val
        self.s4acs.set_cam_config(cam_config)
        sleep(1)
        if not self.s4acs.camera.opmode_err.status:
            self.set_result("error", "The published error msg was not found")
            return

        cam_config[key] = val
        self.s4acs.set_cam_config(cam_config)
        sleep(1)
        if self.s4acs.camera.opmode_err.status:
            self.set_result("error", "The published error msg was not cleaned")
            return


class E014(Test_Strategy):
    _test_code = "E014"

    def run_test(self) -> None:
        limit_values = {
            "TEMP": (-80, 25),
            "COOLER_POWER_STATUS": (0, 1),
        }
        for key, (_min, _max) in limit_values.items():
            time_stamp = datetime.now(timezone.utc)
            self._send_commands_sequence(key, _max, _max + 1)
            self._send_commands_sequence(key, _min, _min - 1)

            lines_list = self.get_log_file_lines()
            filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp)
            filtered_log_lines = self.filter_logs_by_str(filtered_log_lines, "ERROR")
            filtered_log_lines = self.extract_log_msg(filtered_log_lines)

            expected_string = f"The value {_max + 1:.2f} was received for the {key} parameter. However, it should be in the [{_min:.2f}, {_max:.2f}] range."
            self._search_expected_log_msg(filtered_log_lines, expected_string)
            expected_string = f"The value {_min - 1:.2f} was received for the {key} parameter. However, it should be in the [{_min:.2f}, {_max:.2f}] range."
            self._search_expected_log_msg(filtered_log_lines, expected_string)

        return super().run_test()

    def _send_commands_sequence(
        self, key: str, val: float | int, adjusted_val: float | int
    ) -> None:
        self.s4acs.send_command(f"SET {key} {adjusted_val}")
        sleep(1.2)
        if not self.s4acs.camera.opmode_err.status:
            self.set_result("error", "The published error msg was not found")
            return

        self.s4acs.send_command(f"SET {key} {val}")
        sleep(1.2)
        if self.s4acs.camera.opmode_err.status:
            self.set_result("error", "The published error msg was not cleaned")
            return


class E015(Test_Strategy):
    _test_code = "E015"

    def run_test(self) -> None:
        parameters = {
            "ACQUISITION_MODE": (2, 3),
            "TRIGGER_MODE": (1, 0),
            "READ_MODE": (0, 4),
            "VERTICAL_SHIFT_SPEED": (0, 3),
            "VERTICAL_CLOCK_VOLTAGE": (4, 0),
        }
        for key, (forbidden_val, allowed_val) in parameters.items():
            self.s4acs.send_command(f"==== Evaluating {key} parameter ====")
            time_stamp = datetime.now(timezone.utc)
            self._send_commands_sequence(key, forbidden_val, allowed_val)

            lines_list = self.get_log_file_lines()
            filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp)
            filtered_log_lines = self.filter_logs_by_str(filtered_log_lines, "ERROR")
            filtered_log_lines = self.extract_log_msg(filtered_log_lines)

            expected_string = f"The invalid value of {forbidden_val} was found for the {key} parameter."

            self._search_expected_log_msg(filtered_log_lines, expected_string)
            self.s4acs.set_cam_config(self._default_cam_config)

        return super().run_test()

    def _send_commands_sequence(
        self, key: str, forbidden_val: int, allowed_val: int
    ) -> None:
        cam_config = self._default_cam_config.copy()
        cam_config[key] = forbidden_val
        self.s4acs.set_cam_config(cam_config)
        sleep(1)
        if not self.s4acs.camera.opmode_err.status:
            self.set_result("error", "The published error msg was not found")
            return

        cam_config[key] = allowed_val
        self.s4acs.set_cam_config(cam_config)
        sleep(1)
        if self.s4acs.camera.opmode_err.status:
            self.set_result("error", "The published error msg was not cleaned")
            return


class E016(Test_Strategy):
    _test_code = "E016"

    def run_test(self) -> None:
        params_set = [
            (("INITIAL_LINE", 1024), ("FINAL_LINE", 1)),
            (("INITIAL_COLUMN", 1024), ("FINAL_COLUMN", 1)),
            (("INITIAL_COLUMN", 1), ("FINAL_COLUMN", 8), ("HBIN", 3)),
            (("INITIAL_LINE", 1), ("FINAL_LINE", 8), ("VBIN", 3)),
        ]
        for _set in params_set:
            cam_config = self._default_cam_config.copy()
            for param, val in _set:
                cam_config[param] = val
            self.s4acs.set_cam_config(cam_config)
            sleep(1)
            if not self.s4acs.camera.opmode_err.status:
                self.set_result("error", "The published error msg was not found")
                return
        self.s4acs.set_cam_config(self._default_cam_config)
        return super().run_test()


class E017(Test_Strategy):
    _test_code = "E017"

    def run_test(self) -> None:
        limit_values = {
            "WAVEPLATE_POS": (0, 2**16 - 1),
            "#FRAMES": (1, 1500),
            "EXPTIME": (0.00001, 3600),
            "#CYCLES": (1, 10000),
        }
        for key, (_min, _max) in limit_values.items():
            time_stamp = datetime.now(timezone.utc)
            self._send_commands_sequence(key, _max, _max + 1)
            self._send_commands_sequence(key, _min, _min - 1)

            lines_list = self.get_log_file_lines()
            filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp)
            filtered_log_lines = self.filter_logs_by_str(filtered_log_lines, "ERROR")
            filtered_log_lines = self.extract_log_msg(filtered_log_lines)

            expected_string = f"The value {_max + 1:.2f} was received for the {key} parameter. However, it should be in the [{_min:.2f}, {_max:.2f}] range."
            self._search_expected_log_msg(filtered_log_lines, expected_string)
            expected_string = f"The value {_min - 1:.2f} was received for the {key} parameter. However, it should be in the [{_min:.2f}, {_max:.2f}] range."
            self._search_expected_log_msg(filtered_log_lines, expected_string)

        return super().run_test()

    def _send_commands_sequence(
        self, key: str, val: float | int, adjusted_val: float | int
    ) -> None:
        self.s4acs.send_command(f"SET {key} {adjusted_val}")
        sleep(1.2)
        if not self.s4acs.camera.acq_config_err[key].status:
            self.set_result("error", f"{key} - The published error msg was not found")
            return

        self.s4acs.send_command(f"SET {key} {val}")
        sleep(1.2)
        if self.s4acs.camera.acq_config_err[key].status:
            self.set_result("error", f"{key} - The published error msg was not cleaned")
            return


class E018(Test_Strategy):
    _test_code = "E018"

    def run_test(self) -> None:
        wppos = 0b1010101010101010
        s4gui_json = S4GUI_JSON.copy()
        s4gui_json["INSTMODE"] = "POLAR"
        self.framework._external_apps["s4gui"].status = s4gui_json
        self.framework._external_apps["s4gui"].publish_status()
        sleep(1)

        self.s4acs.send_command(f"SET WAVEPLATE_POS {wppos!s}")
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.s4acs.send_command("STOP_ACQUISITION")
        self.wait_return_to_idle()
        sleep(1)
        seqs_number = self.s4acs.camera.received_acq_config.seqs
        if not seqs_number == 8:
            self.set_result("error", f"Unexpected sequences number: {seqs_number}")
        last_image = (
            self.acs_config.image_path / self.s4acs.camera.cam_status.last_image_name
        )
        wpposseq = fits.getheader(last_image)["WPPOSSEQ"]
        if not wpposseq == "2,4,6,8,10,12,14,16":
            self.set_result("error", f"Unexpected wppos sequence: {wpposseq}")

        s4gui_json["INSTMODE"] = "PHOT"
        self.framework._external_apps["s4gui"].status = s4gui_json
        self.framework._external_apps["s4gui"].publish_status()
        self.s4acs.send_command("SET WAVEPLATE_POS 0")
        return super().run_test()


class E020(Test_Strategy):
    _test_code = "E020"

    def run_test(self) -> None:
        s4gui_json = S4GUI_JSON.copy()
        s4gui_json["OBJECT"] = "TEST01"
        self.framework._external_apps["s4gui"].status = s4gui_json
        self.framework._external_apps["s4gui"].publish_status()
        sleep(1)
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        s4gui_json["OBJECT"] = "TEST02"
        self.framework._external_apps["s4gui"].status = s4gui_json
        self.framework._external_apps["s4gui"].publish_status()
        self.wait_return_to_idle()
        sleep(1)
        last_image = (
            self.acs_config.image_path / self.s4acs.camera.cam_status.last_image_name
        )
        object = fits.getheader(last_image)["OBJECT"]
        if object != "TEST01":
            self.set_result("error", f"Unexpected object keyword value: {object}")

        s4gui_json["INSTMODE"] = "Test"
        self.framework._external_apps["s4gui"].status = s4gui_json
        self.framework._external_apps["s4gui"].publish_status()
        return super().run_test()


class E021(Test_Strategy):
    _test_code = "E021"

    def run_test(self) -> None:
        time_stamp_1 = datetime.now(timezone.utc)
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.wait_acquisition_finish()
        sleep(1)

        lines_list = self.get_log_file_lines()
        filtered_log_lines = self.filter_logs_by_timestamp(lines_list, time_stamp_1)
        filtered_log_lines = self.filter_logs_by_str(filtered_log_lines, "DEBUG")
        filtered_log_lines = self.extract_log_msg(filtered_log_lines)

        expected_strings = [
            "The acquisition of the image series has been finished",
        ]
        for _str in expected_strings:
            if _str not in filtered_log_lines:
                self.set_result("error", "Expected log msg was not found")
        return super().run_test()


class E023(Test_Strategy):
    _test_code = "E023"

    def run_test(self) -> None:
        acq_config = self._default_acq_config.copy()
        acq_config["#FRAMES"] = 2
        self.s4acs.set_acquisition_config(acq_config)
        sleep(1)
        cam_config = self._default_cam_config.copy()
        cam_config["READOUT_RATE"] = 1
        self.s4acs.set_cam_config(cam_config)
        sleep(1)
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.wait_acquisition_finish()
        sleep(1)
        current_exp_time = self.s4acs.camera.cam_status.current_exp_time
        if current_exp_time != 2:
            self.set_result("error", f"Unexpected exposure time: {current_exp_time}")
        cam_config["FRAME_TRANSFER"] = True
        self.s4acs.set_cam_config(cam_config)
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.wait_acquisition_finish()
        if current_exp_time != 2:
            self.set_result("error", f"Unexpected exposure time: {current_exp_time}")

        self.s4acs.set_acquisition_config(self._default_acq_config)
        return super().run_test()
