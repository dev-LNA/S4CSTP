import json

import func_tests.data_types as data_types


class Camera:
    def __init__(self) -> None:
        self._requested_cam_config: data_types.Camera_Configuration
        self._received_cam_config: data_types.Camera_Configuration

        self._requested_acq_config: data_types.Acquisition_Configuration
        self._received_acq_config: data_types.Acquisition_Configuration

        self._cam_status: data_types.Camera_Status
        self._comm_status: data_types.Communication_Status
        self._opmode_err: list[dict]

    @property
    def requested_cam_config(self) -> data_types.Camera_Configuration:
        return self._requested_cam_config

    @requested_cam_config.setter
    def requested_cam_config(self, cam_config: dict) -> None:
        self._requested_cam_config = data_types.Camera_Configuration.from_dict(
            cam_config
        )

    @property
    def received_cam_config(self) -> data_types.Camera_Configuration:
        return self._received_cam_config

    @received_cam_config.setter
    def received_cam_config(self, cam_config: dict) -> None:
        self._received_cam_config = data_types.Camera_Configuration.from_dict(
            cam_config
        )

    @property
    def requested_acq_config(self) -> data_types.Acquisition_Configuration:
        return self._requested_acq_config

    @requested_acq_config.setter
    def requested_acq_config(self, acq_config: dict) -> None:
        self._requested_acq_config = data_types.Acquisition_Configuration.from_dict(
            acq_config
        )

    @property
    def received_acq_config(self) -> data_types.Acquisition_Configuration:
        return self._received_acq_config

    @received_acq_config.setter
    def received_acq_config(self, acq_config: dict) -> None:
        self._received_acq_config = data_types.Acquisition_Configuration.from_dict(
            acq_config
        )

    @property
    def cam_status(self) -> data_types.Camera_Status:
        return self._cam_status

    @cam_status.setter
    def cam_status(self, cam_status: dict) -> None:
        self._cam_status = data_types.Camera_Status.from_dict(cam_status)

    @property
    def comm_status(self) -> data_types.Communication_Status:
        return self._comm_status

    @comm_status.setter
    def comm_status(self, comm_status: dict) -> None:
        self._comm_status = data_types.Communication_Status.from_dict(comm_status)

    @property
    def opmode_err(self) -> list:
        return self._opmode_err

    @opmode_err.setter
    def opmode_err(self, opmode_err: list[dict]) -> None:
        self._opmode_err = opmode_err

    def format_cam_config(self) -> str:
        return json.dumps({
            key.upper(): val
            for key, val in self.requested_cam_config.model_dump().items()
        }).replace(" ", "")

    def format_acq_config(self) -> dict[str, str | int | float]:
        _dict = {
            key.upper(): val
            for key, val in self.requested_acq_config.model_dump().items()
        }
        _dict["#CYCLES"] = _dict.pop("CYCLES")
        _dict["#FRAMES"] = _dict.pop("FRAMES")
        _dict["COOLER_POWER_STATUS"] = _dict.pop("COOLER")
        return _dict

    def verify_opmode_err(self) -> bool:
        return len(self._opmode_err) > 0
