from pathlib import Path
from time import sleep

from .test_strategy import Test_Strategy


class I001(Test_Strategy):
    _test_code = "I001"

    def run_test(self) -> None:
        self._s4acs.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.wait_acquisition_finish()
        sleep(2)
        cam_status = self._s4acs.camera.cam_status
        image_name = cam_status.last_image_name
        image_path = self.cfg_file_content.image_path / image_name
        self._validate_channel(image_name)
        self._validate_image_path(image_path)
        self._validate_acs_mode(cam_status.acs_mode)
        self._validate_log_files_folder()
        self._validate_log_level()
        return super().run_test()

    def _validate_channel(self, image_name: str) -> None:
        acs_channel = image_name.split("_")[1]
        if not acs_channel == f"s4c{self.cfg_file_content.channel}":
            self.set_result("error", f"Unexpected instrument channel: {acs_channel}")

    def _validate_acs_mode(self, acs_mode: bool) -> None:
        if acs_mode != self.cfg_file_content.s4acs_mode:
            self.set_result(
                "error",
                f"S4ACS is not in {'real' if self.cfg_file_content.s4acs_mode else 'simulated'} mode.",
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
            if self.cfg_file_content.log_level.name not in file.read():
                self.set_result("error", "Invalid log level")


class I002(Test_Strategy):
    _test_code = "I002"

    def run_test(self) -> None:
        comm_status = self._s4acs.camera.comm_status
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
            log_file_path = self.cfg_file_content.log_file_path / file_name
            if not log_file_path.exists():
                self.set_result("error", f"File {file_name} does not found")
                break
        return super().run_test()


class I005(Test_Strategy):
    _test_code = "I005"

    def run_test(self) -> None:
        self._s4acs.send_command("EXPOSE")
        self.wait_acquisition_finish()
        img_name = self._s4acs.camera.cam_status.last_image_name
        img_idx = img_name.split(".fits")[0].split("_")[2]
        if img_idx != "000011":
            self.set_result("error", f"Unexpected image index: {img_idx}")
        return super().run_test()


class I006(Test_Strategy):
    _test_code = "I006"

    def run_test(self) -> None:
        cam_status = self._s4acs.camera.cam_status
        if not cam_status.power:
            self.set_result("error", "CCD camera is off")

        return super().run_test()
