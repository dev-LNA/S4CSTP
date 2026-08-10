import os
from pathlib import Path
from time import sleep

import func_tests.data_types as data_types
import func_tests.utils as utils

from .test_strategy import Test_Strategy


class I001(Test_Strategy):
    _test_code = "I001"

    def run_test(self) -> None:
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.wait_acquisition_finish()
        sleep(2)
        cam_status = self.s4acs.camera.cam_status
        image_name = cam_status.last_image_name
        image_path = self.acs_config.image_path / image_name
        self._validate_channel(image_name)
        self._validate_image_path(image_path)
        self._validate_acs_mode(cam_status.acs_mode)
        self._validate_log_files_folder()
        self._validate_log_level()
        return super().run_test()

    def _validate_channel(self, image_name: str) -> None:
        acs_channel = image_name.split("_")[1]
        if not acs_channel == f"s4c{self.acs_config.channel}":
            self.set_result("error", f"Unexpected instrument channel: {acs_channel}")

    def _validate_acs_mode(self, acs_mode: bool) -> None:
        if acs_mode != self.acs_config.acs_mode:
            self.set_result(
                "error",
                f"S4ACS is not in {'real' if self.acs_config.acs_mode else 'simulated'} mode.",
            )

    def _validate_image_path(self, image_path: Path) -> None:
        if not image_path.exists():
            self.set_result("error", "Invalid image folder")

    def _validate_log_files_folder(self) -> None:
        if not self.events_log_file.exists():
            self.set_result("error", "Invalid log files folder")

    def _validate_log_level(
        self,
    ) -> None:
        with open(self.events_log_file, "r") as file:
            if self.acs_config.log_level.name not in file.read():
                self.set_result("error", "Invalid log level")


class I002(Test_Strategy):
    _test_code = "I002"

    def run_test(self) -> None:
        comm_status = self.s4acs.camera.comm_status
        for key, val in comm_status.model_dump().items():
            if not val:
                self.set_result(
                    "error", f"There is no communication with {key.upper()}"
                )
                break
        return super().run_test()


class I003(Test_Strategy):
    _test_code = "I003"

    def run_test(self) -> None:
        for file_suffix in ["events"]:
            file_name = self._today_str + "_" + file_suffix + ".log"
            log_file_path = self.acs_config.log_file_path / file_name
            if not log_file_path.exists():
                self.set_result("error", f"File {file_name} does not found")
                break
        return super().run_test()


class I004(Test_Strategy):
    _test_code = "I004"

    def run_test(self) -> None:
        self.s4acs.send_command("STOP_APP")
        sleep(1)

        cfg_file_name = "_acs_config.cfg"
        acs_config = data_types.S4ACS_Config.from_config_file("acs_config.cfg")
        utils.write_cfg_file(acs_config, cfg_file_name)
        acs_config.image_path = acs_config.image_path.parent / "wrong_path"
        utils.write_cfg_file(acs_config)
        utils.run_s4acs_exe()
        if not self.wait_comm(True):
            self.set_result("error", "S4ACS did not initialize")

        if not self.wait_comm(False):
            self.set_result("error", "S4ACS initialized using a wrong path")

        acs_config = data_types.S4ACS_Config.from_config_file("_acs_config.cfg")
        utils.write_cfg_file(acs_config)
        utils.run_s4acs_exe()

        self.wait_cam_on()

        return super().run_test()


class I005(Test_Strategy):
    _test_code = "I005"

    def run_test(self) -> None:
        self.s4acs.send_command("EXPOSE")
        self.wait_acquisition_finish()
        img_name = self.s4acs.camera.cam_status.last_image_name
        img_idx = img_name.split(".fits")[0].split("_")[2]
        if img_idx != "000011":
            self.set_result("error", f"Unexpected image index: {img_idx}")
        return super().run_test()


class I006(Test_Strategy):
    _test_code = "I006"

    def run_test(self) -> None:
        self.s4acs.send_command("STOP_APP")
        sleep(1)

        acs_config = data_types.S4ACS_Config.from_config_file("acs_config.cfg")
        cfg_file_name = "_acs_config.cfg"
        utils.write_cfg_file(acs_config, cfg_file_name)
        acs_config.acs_mode = 1
        utils.write_cfg_file(acs_config)
        utils.run_s4acs_exe()
        if not self.wait_comm(True):
            self.set_result("error", "S4ACS did not initialize")

        if not self.wait_comm(False) and self.acs_config.acs_mode == 1:
            self.set_result("error", "S4ACS initialized without a camera")

        cfg_file_name = "_acs_config.cfg"
        acs_config = data_types.S4ACS_Config.from_config_file(cfg_file_name)
        utils.write_cfg_file(acs_config)
        utils.run_s4acs_exe()

        self.wait_cam_on()

        return super().run_test()


class I007(Test_Strategy):
    _test_code = "I007"

    def run_test(self) -> None:
        self.s4acs.send_command("STOP_APP")
        sleep(1)

        acs_config = data_types.S4ACS_Config.from_config_file("acs_config.cfg")
        cfg_file_name = "_acs_config.cfg"
        utils.write_cfg_file(acs_config, cfg_file_name)
        cfg_file = Path(Path.cwd().root) / "sparc4" / "config" / "acs_config.cfg"
        os.remove(cfg_file)

        utils.run_s4acs_exe()
        if not self.wait_comm(True):
            self.set_result("error", "S4ACS did not initialize")

        if not self.wait_comm(False):
            self.set_result("error", "S4ACS initialized without the cfg file")

        cfg_file_name = "_acs_config.cfg"
        acs_config = data_types.S4ACS_Config.from_config_file(cfg_file_name)
        utils.write_cfg_file(acs_config)
        utils.run_s4acs_exe()

        self.wait_cam_on()

        return super().run_test()
