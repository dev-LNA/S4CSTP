import json
from time import sleep

import func_tests.comm_channel as comm_channel
import func_tests.data_types as data_types
import func_tests.devices as devices
import func_tests.state as state


class Component:  # pragma: no cover
    """
    The Base Component provides the basic functionality of storing a mediator's
    instance inside component objects.
    """

    _delay_time = 0.1

    def __init__(
        self,
        subscriber: comm_channel.Communication_Channel,
        requester: comm_channel.Communication_Channel,
    ) -> None:
        self._subscriber = subscriber
        self._requester = requester
        self._command: data_types.Command

        self._status: dict | None = None
        self._exe_status = data_types.Execution_Status.NONE
        self.camera = devices.Camera()
        self.state: state.State

    @property
    def command(self) -> data_types.Command:
        return self._command

    @command.setter
    def command(self, cmd: data_types.Command) -> None:
        self._command = cmd
        self._command.validate()

    @property
    def status(self) -> dict | None:
        if self._status is None:
            return
        return {
            "cam_config": self.camera.received_cam_config.model_dump(),
            "cam_status": self.camera.cam_status.model_dump(),
            "acq_config": self.camera.received_acq_config.model_dump(),
            "comm_status": self.camera.comm_status.model_dump(),
        }

    @property
    def exe_status(self) -> str:
        return self._exe_status.name

    def get_status_message(self) -> None:
        self._subscriber.receive_msg()
        if self._subscriber.new_msg:
            self._status = json.loads(self._subscriber.received_msg)
        if self._status is None:
            return
        self.camera.received_cam_config = json.loads(self._status["CCD configuration"])
        self.camera.received_acq_config = json.loads(
            self._status["Acquisition configuration"]
        )
        self.camera.cam_status = json.loads(self._status["CCD status"])
        self.camera.comm_status = json.loads(self._status["Communication status"])
        self.camera.opmode_err = json.loads(self._status["WRITE SETUP error"])
        self.update_exe_status()

    def initialize(self) -> None:
        self._subscriber.initialize_comm()
        self._requester.initialize_comm()
        self.transition_to(state.Idle())
        self._exe_status = data_types.Execution_Status.IDLE

    def update_exe_status(self) -> None:
        cam_status = self.camera.cam_status
        acq_config = self.camera.received_acq_config
        if cam_status.status in ["IDLE", "ACQUISITION_ABORTED"]:
            self._exe_status = data_types.Execution_Status.IDLE
            return
        if cam_status.status == "ACTIVE":
            self._exe_status = data_types.Execution_Status.BUSY
            return
        if (
            cam_status.cycles_done < acq_config.cycles
            and cam_status.last_image_name != ""
        ):
            self._exe_status = data_types.Execution_Status.BUSY
            return

    def confirm_command_execution(self) -> None:
        self._command.executed = "on"

    def end(self) -> None:
        self._subscriber.close_comm()
        self._requester.close_comm()

    def transition_to(self, state: state.State) -> None:
        self.state = state
        self.state.component = self

    def send_command(self, _cmmd: str) -> None:
        self.command = data_types.Command(_cmmd)
        self.state.send_command()
        return

    def wait_command_response(self) -> None:
        self._requester.receive_msg()
        return

    def return_comm_status(self) -> bool:
        return self._subscriber.comm_status

    def command_response_received(self) -> bool:
        return self._requester.new_msg

    def reinitialize_requester(self) -> None:
        self._requester.close_comm()
        self._requester.initialize_comm()

    def set_cam_config(self, cam_config: dict) -> None:
        self.camera.requested_cam_config = cam_config
        cmmd = "WRITE_SETUP " + self.camera.format_cam_config()
        self.send_command(cmmd)

    def set_acquisition_config(self, acq_config: dict) -> None:
        self.camera.requested_acq_config = acq_config
        for key, val in self.camera.format_acq_config().items():
            cmmd = "SET " + key.upper() + " " + str(val)
            self.send_command(cmmd)
            sleep(self._delay_time)

    def validate_acq_config(self) -> bool:
        return self.camera.requested_acq_config == self.camera.received_acq_config


class Fake_Component(Component):
    @property
    def status(self) -> dict | None:
        return None

    def initialize(self) -> None:
        pass

    def get_status_message(self) -> None:
        return

    def update_exe_status(self) -> None:
        self._exe_status = data_types.Execution_Status.IDLE

    def end(self) -> None:
        return super().end()

    def confirm_command_execution(self) -> None:
        return super().confirm_command_execution()

    def return_comm_status(self) -> bool:
        return True

    def send_command(self, _cmmd: str) -> None:
        return
