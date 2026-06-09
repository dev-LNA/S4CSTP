import src.data_types as data_types


class Optic_Fiber:
    def __init__(
        self,
        name: str,
        allowed_light_srcs: data_types.Light_Sources,
    ) -> None:
        self._name = name
        self._light_source = data_types.Light_Sources.NONE
        self.allowed_light_srcs = allowed_light_srcs

    @property
    def name(self) -> str:
        return self._name

    @property
    def light_source(self) -> data_types.Light_Sources:
        return self._light_source

    @light_source.setter
    def light_source(self, light_source: data_types.Light_Sources) -> None:
        if light_source not in self.allowed_light_srcs:
            raise ValueError(
                f"Provided options not in allowed light sources: {light_source}"
            )
        self._light_source = light_source
        return
