from abc import ABC, abstractmethod

import func_tests.component as component
import func_tests.data_types as data_types


class Test_Strategy(ABC):
    def __init__(self, component: component.Component) -> None:
        self._component = component
        self._commands_list: list[data_types.Command]
        self._result = data_types.Test_Result

    @abstractmethod
    def run_test() -> data_types.Test_Result: ...
