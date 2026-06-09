import logging
from abc import ABC
from datetime import datetime, timedelta

import src.component as component
import src.data_types as data_types


class State(ABC):  # pragma: no cover
    timeout_time = 1  # sec

    @property
    def component(self) -> component.Component:
        return self._component

    @component.setter
    def component(self, component: component.Component) -> None:
        self._component = component

    def initialize(self) -> None:
        raise RuntimeError(
            f"You cannot initialize component in {type(self).__name__} state."
        )

    def handle_command(self) -> None:
        raise RuntimeError(f"You cannot handle command in {type(self).__name__} state.")

    def send_command(self) -> None:
        raise RuntimeError(f"You cannot send command in {type(self).__name__} state.")

    def receive_command_response(self) -> None:
        raise RuntimeError(
            f"You cannot wait command response in {type(self).__name__} state."
        )

    def supports_command(self) -> None:
        raise RuntimeError(
            f"You cannot verify if the command is supported in {type(self).__name__} state."
        )

    def handle_invalid_command(self) -> None:
        raise RuntimeError(
            f"You cannot handle invalid command in {type(self).__name__} state."
        )

    def handle_unsupported_command(self) -> None:
        raise RuntimeError(
            f"You cannot handle unsupported command in {type(self).__name__} state."
        )

    def handle_time_out(self) -> None:
        raise RuntimeError(
            f"You cannot handle time out in {type(self).__name__} state."
        )

    def handle_busy_componnent(self) -> None:  # TODO: tratar isso melhor
        raise RuntimeError(
            f"Control system is not in the Idle state: {type(self).__name__}."
        )

    def return_to_idle(self) -> None:
        raise RuntimeError(f"You cannot return to Idle in {type(self).__name__} state.")


class Not_Initialized(State):
    def initialize(self) -> None:
        self.component.initialize()
        self.component.transition_to(Idle())


class Idle(State):
    def handle_command(self) -> None:
        if not self.component.command.valid:
            self.component.state.handle_invalid_command()
            return
        logging.debug("The command is valid")
        self.component.transition_to(Command_Validated())
        self.component.state.supports_command()

    def handle_invalid_command(self) -> None:
        command = self.component.command
        logging.warning(f"Invalid command {command.str}")
        self.component.transition_to(Idle())


class Command_Validated(State):
    def supports_command(self) -> None:
        supported = self.component.supports_command()
        if not supported:
            self.component.state.handle_unsupported_command()
            return
        logging.debug("The command is supported")
        self.component.command.supported = data_types.Led_Status.ON
        self.component.transition_to(Supported_Command())
        self.component.state.send_command()

    def handle_unsupported_command(self) -> None:
        logging.warning(
            f"Unsupported command: {self.component.command._dict['field1']}."
        )
        self.component.command.supported = data_types.Led_Status.ERROR
        self.component.transition_to(Idle())


class Supported_Command(State):
    def send_command(self) -> None:
        self.component.send_request(self.component.command.str)
        logging.debug("The command was sent")
        self.component.transition_to(Command_Sent())
        self.component._state.receive_command_response()


class Command_Sent(State):
    def receive_command_response(self) -> None:
        time_stamp1, time_interv = datetime.now(), timedelta()
        while time_interv.seconds < self.timeout_time:
            self.component.wait_command_response()
            if self.component.command_response_received():
                self.component.command.timeout = data_types.Led_Status.ON
                logging.debug("The command response was received")
                self.component.transition_to(Command_Response_Received())
                self.component.state.return_to_idle()
                return
            time_interv = datetime.now() - time_stamp1
        self.component.state.handle_time_out()

    def handle_time_out(self) -> None:
        command = self.component.command
        self.component.reinitialize_requester()
        self.component.command.timeout = data_types.Led_Status.ERROR
        logging.warning(f"Unanswered command: {command._dict['field1']}.")
        self.component.transition_to(Idle())


class Command_Response_Received(State):
    def return_to_idle(self) -> None:
        self.component.transition_to(Idle())
        return
