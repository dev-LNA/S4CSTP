import logging
from abc import ABC, abstractmethod
from time import sleep

import func_tests.component as component
import func_tests.data_types as data_types


class Test_Strategy(ABC):
    def __init__(self) -> None:
        self._component: component.Component
        self._commands_list: list[data_types.Command]
        self.result: data_types.Test_Result | None
        self._test_code = "F000"

    @abstractmethod
    def run_test(self) -> None: ...

    def set_component(self, component: component.Component) -> None:
        self._component = component


class Fake_Test(Test_Strategy):
    def run_test(self) -> None:
        logging.info(f"Running test {self._test_code}...")
        sleep(0.5)
        self.result = data_types.Test_Result(
            success=True, test_code=self._test_code, message="Done"
        )
        logging.debug(f"Result: {self.result.model_dump()}")
        return
