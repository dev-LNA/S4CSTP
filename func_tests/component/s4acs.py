import json

import func_tests.comm_channel as comm_channel
import func_tests.component as component
import func_tests.data_types as data_types


class S4ACS(component.Component):
    def __init__(
        self,
        subscriber: comm_channel.Communication_Channel,
        requester: comm_channel.Communication_Channel,
    ) -> None:
        super().__init__(subscriber, requester)
        self._allowed_commands = [
            "EXPOSE",
            "SET",
            "STOP_APP",
            "WRITE_SETUP",
            "WAIT_EXPOSE_COMMAND",
            "ABORT_ACQUISITION",
            "PAUSE_ACQUISITION",
            "RESUME_ACQUISITION",
        ]

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
            "comm_status": self.camera.comm_status.model_dump(),
        }

    def get_status_message(self) -> None:
        super().get_status_message()
        if self._status is None:
            return
        self.camera.cam_config = json.loads(self._status["CCD configuration"])
        self.camera.acq_config = json.loads(self._status["Acquisition configuration"])
        self.camera.cam_status = json.loads(self._status["CCD status"])
        self.camera.comm_status = json.loads(self._status["Communication status"])
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
