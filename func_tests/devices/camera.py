import json
from pathlib import Path

import pandas as pd

import func_tests.data_types as data_types


class Camera:
    ACQUISITION_CONFIG_PARAMETERS = [
        "WAVEPLATE_POS",
        "SUFFIX",
        "#CYCLES",
        "#FRAMES",
        "COOLER_POWER_STATUS",
        "EXPTIME",
        "TEMP",
    ]

    def __init__(self) -> None:
        self._requested_cam_config: data_types.Camera_Configuration
        self._received_cam_config: data_types.Camera_Configuration

        self._requested_acq_config: data_types.Acquisition_Configuration
        self._received_acq_config: data_types.Acquisition_Configuration

        self._cam_status: data_types.Camera_Status
        self._comm_status: data_types.Communication_Status
        self._opmode_err: data_types.Error_Type
        self._acq_config_err: dict[str, data_types.Error_Type] = {
            key: data_types.Error_Type(status=False, code=0, source="")
            for key in self.ACQUISITION_CONFIG_PARAMETERS
        }
        self.opmode_params_limits: dict
        self.get_opmode_param_limits()

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
    def opmode_err(self) -> data_types.Error_Type:
        return self._opmode_err

    @opmode_err.setter
    def opmode_err(self, opmode_err: dict) -> None:
        self._opmode_err = data_types.Error_Type.from_dict(opmode_err)

    @property
    def acq_config_err(self) -> dict[str, data_types.Error_Type]:
        return self._acq_config_err

    @acq_config_err.setter
    def acq_config_err(self, acq_config_err: dict) -> None:
        parameters = acq_config_err.keys()
        if self.ACQUISITION_CONFIG_PARAMETERS.sort() != list(parameters).sort():
            raise ValueError(f"Unexpected set of parameters: {parameters}")
        for key, val in acq_config_err.items():
            self._acq_config_err[key] = data_types.Error_Type.from_dict(val)

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

    def return_acquisition_error(self) -> bool:
        return True in [val.status for val in self._acq_config_err.values()]

    def convert_acq_cfg_err_to_dict(self) -> dict:
        return {key: val.model_dump() for key, val in self._acq_config_err.items()}

    def get_opmode_param_limits(self) -> None:
        file_path = (
            Path.cwd() / "func_tests" / "_csv" / "cam_parameters_limit_values.csv"
        )
        df = pd.read_csv(file_path)
        self.opmode_params_limits = dict(
            zip(df["parameter"], zip(df["lower limit"], df["upper limit"]))
        )
