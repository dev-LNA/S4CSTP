from datetime import datetime, timezone
from time import sleep

import func_tests.data_types as data_types

from .test_strategy import Test_Strategy


class S001(Test_Strategy):
    _test_code = "S001"

    def run_test(self) -> None:
        self._default_acq_config["COOLER_POWER_STATUS"] = 1
        self._s4acs.set_acquisition_config(self._default_acq_config)
        self.validate_acq_config()
        self._s4acs.send_command("STOP_APP")
        sleep(2)
        self._default_acq_config["COOLER_POWER_STATUS"] = 0
        self._s4acs.camera.requested_acq_config = self._default_acq_config
        if not self._s4acs.validate_acq_config():
            self.set_result("error", "Unexpected acquisition configuration")

        return super().run_test()
