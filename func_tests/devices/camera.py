import json

import func_tests.data_types as data_types


class Camera:
    def __init__(self) -> None:
        self._cam_config: data_types.Camera_Configuration
        self._acq_config: data_types.Acquisition_Configuration
        self._cam_status: data_types.Camera_Status
        self._comm_status: data_types.Communication_Status
        self._opmode_err: list[dict]

    @property
    def cam_config(self) -> data_types.Camera_Configuration:
        return self._cam_config

    @property
    def acq_config(self) -> data_types.Acquisition_Configuration:
        return self._acq_config

    @property
    def cam_status(self) -> data_types.Camera_Status:
        return self._cam_status

    @property
    def comm_status(self) -> data_types.Communication_Status:
        return self._comm_status

    @property
    def opmode_err(self) -> list:
        return self._opmode_err

    @cam_config.setter
    def cam_config(self, cam_config: dict) -> None:
        self._cam_config = data_types.Camera_Configuration.from_dict(cam_config)

    @acq_config.setter
    def acq_config(self, acq_config: dict) -> None:
        self._acq_config = data_types.Acquisition_Configuration.from_dict(acq_config)

    @cam_status.setter
    def cam_status(self, cam_status: dict) -> None:
        self._cam_status = data_types.Camera_Status.from_dict(cam_status)

    @comm_status.setter
    def comm_status(self, comm_status: dict) -> None:
        self._comm_status = data_types.Communication_Status.from_dict(comm_status)

    @opmode_err.setter
    def opmode_err(self, opmode_err: list[dict]) -> None:
        self._opmode_err = opmode_err

    def return_formatted_config(self) -> str:
        return json.dumps({
            key.upper(): val for key, val in self.cam_config.model_dump().items()
        }).replace(" ", "")

    def verify_opmode_err(self) -> bool:
        return len(self._opmode_err) > 0
