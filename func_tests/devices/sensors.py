from abc import ABC, abstractmethod

import src.data_types as data_types


class Sensor(ABC):  # pragma: no cover
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self._level = data_types.Level.NONE
        self._current_val = None

    @property
    def current_val(self) -> float:
        if self._current_val is None:
            raise ValueError("The current value was not initialized.")
        return self._current_val

    @current_val.setter
    def current_val(self, new_val: float) -> None:
        self._current_val = new_val

    @property
    def level(self) -> data_types.Level:
        return self._level

    @abstractmethod
    def update_level(self) -> None:
        pass


class Current_Sensor(Sensor):
    def update_level(self) -> None:
        return super().update_level()


class Temperature_Sensor(Sensor):
    def update_level(self) -> None:
        return super().update_level()
