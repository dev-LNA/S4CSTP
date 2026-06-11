import logging
from abc import ABC, abstractmethod
from time import sleep

import func_tests.component as component
import func_tests.data_types as data_types


class Test_Strategy(ABC):
    _test_code = "A000"

    def __init__(self) -> None:
        self._component: component.Component
        self._commands_list: list[data_types.Command]
        self.result: data_types.Test_Result | None = None

    @abstractmethod
    def run_test(self) -> None:
        logging.info(f"Running test {self._test_code}...")
        if self.result is not None:
            logging.debug(f"Result: {self.result.model_dump()}")

    def set_component(self, component: component.Component) -> None:
        self._component = component


class Fake_Positive_Test(Test_Strategy):
    _test_code = "P000"

    def run_test(self) -> None:
        sleep(0.5)
        self.result = data_types.Test_Result(
            success="on", test_code=self._test_code, message="Done"
        )
        return super().run_test()


class Fake_Negative_Test(Test_Strategy):
    _test_code = "N000"

    def run_test(self) -> None:
        sleep(0.5)
        self.result = data_types.Test_Result(
            success="error", test_code=self._test_code, message="Done"
        )
        return super().run_test()
