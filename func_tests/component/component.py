import json
from abc import ABC, abstractmethod
from time import sleep

import func_tests.comm_channel as comm_channel
import func_tests.data_types as data_types
import func_tests.devices as devices
import func_tests.state as state


class Component(ABC):  # pragma: no cover
    """
    The Base Component provides the basic functionality of storing a mediator's
    instance inside component objects.
    """

    _gui_std_delay = 0.2

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
        self._allowed_commands = []
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
        return self._status

    @property
    def exe_status(self) -> str:
        return self._exe_status.name

    @abstractmethod
    def get_status_message(self) -> None:
        self._subscriber.receive_msg()
        if self._subscriber.new_msg:
            self._status = json.loads(self._subscriber.received_msg)

    def send_command(self, _cmmd: str) -> None:
        self.command = data_types.Command(_cmmd)
        self.state.send_command()
        return

    def wait_command_response(self) -> None:
        self._requester.receive_msg()
        return

    @abstractmethod
    def initialize(self) -> None:
        self._subscriber.initialize_comm()
        self._requester.initialize_comm()
        self.transition_to(state.Idle())
        self._exe_status = data_types.Execution_Status.IDLE

    @abstractmethod
    def update_exe_status(self) -> None:
        self._exe_status = data_types.Execution_Status.IDLE

    @abstractmethod
    def confirm_command_execution(self) -> None:
        self._command.executed = "on"

    @abstractmethod
    def end(self) -> None:
        self._subscriber.close_comm()
        self._requester.close_comm()

    def transition_to(self, state: state.State) -> None:
        self.state = state
        self.state.component = self

    def return_comm_status(self) -> bool:
        return self._subscriber.comm_status

    def command_response_received(self) -> bool:
        return self._requester.new_msg

    def reinitialize_requester(self) -> None:
        self._requester.close_comm()
        self._requester.initialize_comm()

    def set_cam_config(self, cam_config: dict) -> None:
        self.camera.cam_config = cam_config
        cmmd = "WRITE_SETUP " + self.camera.return_formatted_cam_config()
        self.send_command(cmmd)

    def set_acquisition_config(self, acq_config: dict) -> None:
        self.camera.acq_config = acq_config
        for key, val in self.camera.return_formatted_acq_config().items():
            cmmd = "SET " + key.upper() + " " + str(val)
            self.send_command(cmmd)
            sleep(self._gui_std_delay)


class Fake_Component(Component):
    def initialize(self) -> None:
        return super().initialize()

    def get_status_message(self) -> None:
        return super().get_status_message()

    def update_exe_status(self) -> None:
        return super().update_exe_status()

    def end(self) -> None:
        return super().end()

    def confirm_command_execution(self) -> None:
        return super().confirm_command_execution()
