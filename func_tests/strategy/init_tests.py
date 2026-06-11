import func_tests.data_types as data_types

from .test_strategy import Test_Strategy


class I006(Test_Strategy):
    _test_code = "I006"

    def run_test(self) -> None:
        if self._component.status is not None:
            cam_status = self._component.camera.cam_status
            if cam_status.power:
                self.result = data_types.Test_Result(
                    success="on", test_code=self._test_code, message="Done"
                )
            else:
                self.result = data_types.Test_Result(
                    success="error",
                    test_code=self._test_code,
                    message="CCD camera is not on.",
                )
        return super().run_test()
