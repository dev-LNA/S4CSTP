import func_tests.data_types as data_types


class Camera:
    def __init__(self) -> None:
        self._cam_config: data_types.Camera_Configuration
        self._acq_config: data_types.Acquisition_Configuration
        self._cam_status: data_types.Camera_Status

    @property
    def cam_config(self) -> data_types.Camera_Configuration:
        return self._cam_config

    @property
    def acq_config(self) -> data_types.Acquisition_Configuration:
        return self._acq_config

    @property
    def cam_status(self) -> data_types.Camera_Status:
        return self._cam_status

    @cam_config.setter
    def cam_config(self, cam_config: dict) -> None:
        self._cam_config = data_types.Camera_Configuration.from_dict(cam_config)

    @acq_config.setter
    def acq_config(self, acq_config: dict) -> None:
        self._acq_config = data_types.Acquisition_Configuration.from_dict(acq_config)

    @cam_status.setter
    def cam_status(self, cam_status: dict) -> None:
        self._cam_status = data_types.Camera_Status.from_dict(cam_status)
