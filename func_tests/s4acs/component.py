import json
from abc import ABC, abstractmethod

import src.comm_protocol as comm_protocol
import src.data_types as data_types
import src.mediator as mediator
import src.state.component as comp_state


class Component(ABC):  # pragma: no cover
    """
    The Base Component provides the basic functionality of storing a mediator's
    instance inside component objects.
    """

    def __init__(
        self,
        mediator: mediator.Mediator | None,
        subscriber: comm_protocol.Communication_Protocol,
        requester: comm_protocol.Communication_Protocol,
    ) -> None:
        self._subscriber = subscriber
        self._requester = requester
        self._state: comp_state.State
        self._mediator = mediator

        self._status: dict | None = None
        self._exe_status = data_types.Execution_Status.NONE
        self._command: data_types.Command
        self._allowed_commands = []
        self.transition_to(comp_state.Not_Initialized())

    @property
    def mediator(self) -> mediator.Mediator:
        if self._mediator is None:
            raise RuntimeError("Mediator not initialized.")
        return self._mediator

    @mediator.setter
    def mediator(self, mediator: mediator.Mediator) -> None:
        self._mediator = mediator

    @property
    def status(self) -> dict | None:
        return self._status

    @property
    def state(self) -> comp_state.State:
        return self._state

    @property
    def exe_status(self) -> str:
        return self._exe_status.name

    @property
    def command(self) -> data_types.Command:
        return self._command

    @command.setter
    def command(self, new_command: data_types.Command) -> None:
        self._command = new_command
        self._command.validate()
        return

    def transition_to(self, state: comp_state.State) -> None:
        self._state = state
        self._state.component = self

    @abstractmethod
    def get_status_message(self) -> None:
        self._subscriber.receive_msg()
        if self._subscriber._new_msg:
            self._status = json.loads(self._subscriber.received_msg)

    def send_request(self, request: str) -> None:
        self._requester.send_msg(request)
        return

    def wait_command_response(self) -> None:
        self._requester.receive_msg()
        return

    @abstractmethod
    def initialize(self) -> None:
        self._subscriber.initialize_comm()
        self._requester.initialize_comm()
        self._exe_status = data_types.Execution_Status.IDLE

    @abstractmethod
    def update_exe_status(self) -> None:
        self._exe_status = data_types.Execution_Status.IDLE

    @abstractmethod
    def confirm_command_execution(self) -> None:
        self.command.executed = "on"

    @abstractmethod
    def end(self) -> None:
        self._subscriber.close_comm()
        self._requester.close_comm()

    def return_comm_status(self) -> bool:
        return self._subscriber.comm_status

    def command_response_received(self) -> bool:
        return self._requester.new_msg

    def reinitialize_requester(self) -> None:
        self._requester.close_comm()
        self._requester.initialize_comm()

    def supports_command(self) -> bool:
        return self.command._dict["field1"] in self._allowed_commands


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
