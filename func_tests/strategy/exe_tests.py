import json

import func_tests.data_types as data_types

from .test_strategy import Test_Strategy


class E001(Test_Strategy):
    _test_code = "E001"

    def run_test(self) -> None:
        delay = self.wait_2_pub_msgs()
        if delay.seconds < 1:
            self.set_result("error", "Interval between pub msgs smaller than 1 s")
        self._component.state.send_command("EXPOSE")
        self.wait_acquisition_start()
        delay = self.wait_2_pub_msgs()
        self.wait_return_to_idle()
        if delay.seconds > 0.2:
            self.set_result("error", "Interval between pub msgs larger than 0.2 s")
        self.set_result("on", "Done")
        return super().run_test()


class E003(Test_Strategy):
    _test_code = "E003"

    def run_test(self) -> None:
        self._default_cam_config["INITIAL_LINE"] = 1025
        self._component.set_cam_config(self._default_cam_config)

        with open(self.events_log_file) as file:
            file_content = file.read()
        for level in data_types.Log_Level:
            if level.name == "CRITICAL":  # TODO: adicionar um commando para isso
                continue
            if self.acs_log_level.value > level.value:
                continue
            if level.name not in file_content:
                self.set_result("error", f"Log level {level.name} not found")

        self._default_cam_config["INITIAL_LINE"] = 1024
        self._component.set_cam_config(self._default_cam_config)
        self.set_result("on", "Done")

        return super().run_test()
