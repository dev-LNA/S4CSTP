from pathlib import Path

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
        if delay.seconds > 0.2:
            self.set_result("error", "Interval between pub msgs larger than 0.2 s")
        self.set_result("on", "Done")
        return super().run_test()
