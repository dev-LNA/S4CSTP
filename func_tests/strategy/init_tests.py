from pathlib import Path
from time import sleep

from .test_strategy import Test_Strategy


class I001(Test_Strategy):
    _test_code = "I001"

    def run_test(self) -> None:
        self._component.send_command("EXPOSE")
        self.wait_acquisition_start()
        self.wait_acquisition_finish()
        sleep(2)
        cam_status = self._component.camera.cam_status
        image_name = cam_status.last_image_name
        image_path = self.imgs_folder / image_name
        self._validate_channel(image_name)
        self._validate_image_path(image_path)
        self._validate_acs_mode(cam_status.acs_mode)
        self._validate_log_files_folder()
        self._validate_log_level()
        return super().run_test()

    def _validate_channel(self, image_name: str) -> None:
        acs_channel = image_name.split("_")[1]
        if not acs_channel == f"s4c{self.channel}":
            self.set_result("error", f"Unexpected instrument channel: {acs_channel}")

    def _validate_acs_mode(self, acs_mode: bool) -> None:
        if acs_mode != self.acs_mode:
            self.set_result(
                "error",
                f"S4ACS is not in {'real' if self.acs_mode else 'simulated'} mode.",
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
            if self.acs_log_level.name not in file.read():
                self.set_result("error", "Invalid log level")


class I002(Test_Strategy):
    _test_code = "I002"

    def run_test(self) -> None:
        comm_status = self._component.camera.comm_status
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
            log_file_path = self.log_folder / file_name
            if not log_file_path.exists():
                self.set_result("error", f"File {file_name} does not found")
                break
        return super().run_test()


class I006(Test_Strategy):
    _test_code = "I006"

    def run_test(self) -> None:
        cam_status = self._component.camera.cam_status
        if not cam_status.power:
            self.set_result("error", "CCD camera is off")

        return super().run_test()
