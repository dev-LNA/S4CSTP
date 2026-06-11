import logging
from abc import ABC
from datetime import datetime, timedelta

import func_tests.component as component
import func_tests.data_types as data_types


class State(ABC):  # pragma: no cover
    timeout_time = 1  # sec

    @property
    def component(self) -> component.Component:
        return self._component

    @component.setter
    def component(self, component: component.Component) -> None:
        self._component = component

    def send_command(self) -> None:
        raise RuntimeError(f"You cannot send command in {type(self).__name__} state.")

    def receive_command_response(self) -> None:
        raise RuntimeError(
            f"You cannot wait command response in {type(self).__name__} state."
        )

    def handle_time_out(self) -> None:
        raise RuntimeError(
            f"You cannot handle time out in {type(self).__name__} state."
        )

    def return_to_idle(self) -> None:
        raise RuntimeError(f"You cannot return to Idle in {type(self).__name__} state.")


class Idle(State):
    def send_command(self) -> None:
        cmmd = self.component.command.str
        self.component.send_request(cmmd)
        logging.debug(f"The command {cmmd} was sent")
        self.component.transition_to(Command_Sent())
        self.component.state.receive_command_response()


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
