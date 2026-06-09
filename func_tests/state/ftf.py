import logging
from abc import ABC

import func_tests.data_types as data_types
import func_tests.ftf as ftf


class State(ABC):  # pragma: no cover
    timeout_time = 1  # sec

    @property
    def framework(self) -> ftf.Functionalities_Tests_Framework:
        return self._framework

    @framework.setter
    def framework(self, framework: ftf.Functionalities_Tests_Framework) -> None:
        self._framework = framework

    def initialize(self) -> None:
        raise RuntimeError(
            f"You cannot initialize framework in {type(self).__name__} state."
        )

    def receive_client_request(self) -> None:
        raise RuntimeError(
            f"You cannot receive client request in {type(self).__name__} state."
        )

    def validate_request(self) -> None:
        raise RuntimeError(
            f"You cannot validate request in {type(self).__name__} state."
        )

    def dispatch_command(self) -> None:
        raise RuntimeError(
            f"You cannot dispatch command in {type(self).__name__} state."
        )

    def handle_invalid_request(self) -> None:
        raise RuntimeError(
            f"You cannot handle invalid request in {type(self).__name__} state."
        )

    def handle_invalid_recipient(self) -> None:
        raise RuntimeError(
            f"You cannot handle invalid recipient in {type(self).__name__} state."
        )

    def verify_request_recipient(self) -> None:
        raise RuntimeError(
            f"You cannot verify the request recipient in {type(self).__name__} state."
        )

    def verify_component_availability(self) -> None:
        raise RuntimeError(
            f"You cannot verify the component availability in {type(self).__name__} state."
        )

    def handle_busy_framework(self) -> None:  # TODO: tratar isso melhor
        raise RuntimeError(
            f"Control system is not in the Idle state: {type(self).__name__}."
        )

    def handle_not_idle_component(self) -> None:
        raise RuntimeError(
            f"You cannot handle busy component in Idle state: {type(self).__name__}."
        )

    def return_to_idle(self) -> None:
        raise RuntimeError(f"You cannot return to Idle in {type(self).__name__} state.")


class Not_Initialized(State):
    def initialize(self) -> None:
        self.framework.create_log_file()
        logging.info("Framework was started")
        self.framework.s4acs.initialize()
        self.framework.transition_to(Idle())
        logging.debug("Framework was initialized succesfully")


class Idle(State):
    def receive_client_request(self) -> None:
        for _component in self.framework.container.values:
            _component.command = data_types.Command("")
        self.framework.transition_to(Request_Received())
        self.framework._state.validate_request()
        return


class Request_Received(State):
    def validate_request(self) -> None:
        request = self.framework.client_request
        if request.valid:
            logging.debug("The request is valid")
            request.status = data_types.Led_Status.ON
            self.framework.transition_to(Request_Validated())
            self.framework._state.verify_request_recipient()
            return
        self.framework._state.handle_invalid_request()

    def handle_invalid_request(self) -> None:
        request = self.framework.client_request
        request.status = data_types.Led_Status.ERROR
        logging.warning(f"Invalid request {request.str}")
        self.framework.transition_to(Idle())


class Request_Validated(State):
    def verify_request_recipient(self) -> None:
        request = self.framework.client_request
        valid_recipient = request._dict["field1"] in self.framework.container.keys
        if valid_recipient:
            logging.debug("The recipient is valid")
            request.recipient = data_types.Led_Status.ON
            self.framework.transition_to(Recipient_Validated())
            self.framework._state.verify_component_availability()
            return
        self.framework._state.handle_invalid_recipient()

    def handle_invalid_recipient(self) -> None:
        request = self.framework.client_request
        request.recipient = data_types.Led_Status.ERROR
        logging.warning(f"Invalid recipient {request.str}")
        self.framework.transition_to(Idle())


class Recipient_Validated(State):
    def verify_component_availability(self) -> None:
        request = self.framework.client_request
        recipient = request._dict["field1"]
        _component = self.framework.container._dict[recipient]
        if _component.exe_status != "IDLE":
            self.handle_not_idle_component()
            return
        self.framework.transition_to(Component_Available())
        self.framework._state.dispatch_command()

    def handle_not_idle_component(self) -> None:
        request = self.framework.client_request
        recipient = request._dict["field1"]
        request.recipient = data_types.Led_Status.WARNING
        logging.warning(f"{recipient} is not in the IDLE execution state")
        self.framework.transition_to(Idle())


class Component_Available(State):
    def dispatch_command(self) -> None:
        request = self.framework.client_request
        recipient = request._dict["field1"]
        _component = self.framework.container._dict[recipient]
        _component.command = data_types.Command(request.command)
        _component.state.handle_command()

        self.framework.transition_to(Command_Dispatched())
        self.framework._state.return_to_idle()


class Command_Dispatched(State):
    def return_to_idle(self) -> None:
        self.framework.transition_to(Idle())
        return
