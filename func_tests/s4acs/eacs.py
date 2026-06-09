import json

import src.comm_protocol as comm_protocol
import src.component as comp
import src.data_types as data_types
import src.mediator as mediator
from src.devices import Camera


class EACS(comp.Component):
    def __init__(
        self,
        mediator: mediator.Mediator | None,
        subscriber: comm_protocol.Communication_Protocol,
        requester: comm_protocol.Communication_Protocol,
    ) -> None:
        super().__init__(mediator, subscriber, requester)
        self._allowed_commands = [  # criar uma classe Command?
            "EXPOSE",
            "SET",
            "STOP_APP",
            "WRITE_SETUP",
            "WAIT_EXPOSE_COMMAND",
            "ABORT_ACQUISITION",
            "PAUSE_ACQUISITION",
            "RESUME_ACQUISITION",
        ]
        self.camera = Camera()

    def initialize(self) -> None:
        return super().initialize()

    @property
    def status(self) -> dict | None:
        if self._status is None:
            return
        return {
            "cam_config": self.camera.cam_config.model_dump(),
            "cam_status": self.camera.cam_status.model_dump(),
            "acq_config": self.camera.acq_config.model_dump(),
        }

    def get_status_message(self) -> None:
        super().get_status_message()
        if self._status is None:
            return
        self.camera.cam_config = json.loads(self._status["CCD configuration"])
        self.camera.acq_config = json.loads(self._status["Acquisition configuration"])
        self.camera.cam_status = json.loads(self._status["CCD status"])
        self.update_exe_status()

    def end(self) -> None:
        return super().end()

    def update_exe_status(self) -> None:
        cam_status = self.camera.cam_status
        acq_config = self.camera.acq_config
        if (
            cam_status.cycles_done < acq_config.cycles
            and cam_status.last_image_name != ""
        ):
            self._exe_status = data_types.Execution_Status.BUSY
            return
        self._exe_status = data_types.Execution_Status.IDLE

    def confirm_command_execution(self) -> None:
        return super().confirm_command_execution()
