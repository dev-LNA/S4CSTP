import logging
from abc import ABC

import src.data_types as data_types
import src.mediator as mediator


class State(ABC):  # pragma: no cover
    timeout_time = 1  # sec

    @property
    def mediator(self) -> mediator.EMCS:
        return self._mediator

    @mediator.setter
    def mediator(self, mediator: mediator.EMCS) -> None:
        self._mediator = mediator

    def initialize(self) -> None:
        raise RuntimeError(
            f"You cannot initialize mediator in {type(self).__name__} state."
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

    def handle_busy_mediator(self) -> None:  # TODO: tratar isso melhor
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
        self.mediator.create_log_file()
        logging.info("EMCS was started")
        for component in self.mediator._container.values:
            component.state.initialize()
        self.mediator.transition_to(Idle())
        logging.debug("EMCS was initialized succesfully")


class Idle(State):
    def receive_client_request(self) -> None:
        for _component in self.mediator.container.values:
            _component.command = data_types.Command("")
        self.mediator.transition_to(Request_Received())
        self.mediator._state.validate_request()
        return


class Request_Received(State):
    def validate_request(self) -> None:
        request = self.mediator.client_request
        if request.valid:
            logging.debug("The request is valid")
            request.status = data_types.Led_Status.ON
            self.mediator.transition_to(Request_Validated())
            self.mediator._state.verify_request_recipient()
            return
        self.mediator._state.handle_invalid_request()

    def handle_invalid_request(self) -> None:
        request = self.mediator.client_request
        request.status = data_types.Led_Status.ERROR
        logging.warning(f"Invalid request {request.str}")
        self.mediator.transition_to(Idle())


class Request_Validated(State):
    def verify_request_recipient(self) -> None:
        request = self.mediator.client_request
        valid_recipient = request._dict["field1"] in self.mediator.container.keys
        if valid_recipient:
            logging.debug("The recipient is valid")
            request.recipient = data_types.Led_Status.ON
            self.mediator.transition_to(Recipient_Validated())
            self.mediator._state.verify_component_availability()
            return
        self.mediator._state.handle_invalid_recipient()

    def handle_invalid_recipient(self) -> None:
        request = self.mediator.client_request
        request.recipient = data_types.Led_Status.ERROR
        logging.warning(f"Invalid recipient {request.str}")
        self.mediator.transition_to(Idle())


class Recipient_Validated(State):
    def verify_component_availability(self) -> None:
        request = self.mediator.client_request
        recipient = request._dict["field1"]
        _component = self.mediator.container._dict[recipient]
        if _component.exe_status != "IDLE":
            self.handle_not_idle_component()
            return
        self.mediator.transition_to(Component_Available())
        self.mediator._state.dispatch_command()

    def handle_not_idle_component(self) -> None:
        request = self.mediator.client_request
        recipient = request._dict["field1"]
        request.recipient = data_types.Led_Status.WARNING
        logging.warning(f"{recipient} is not in the IDLE execution state")
        self.mediator.transition_to(Idle())


class Component_Available(State):
    def dispatch_command(self) -> None:
        request = self.mediator.client_request
        recipient = request._dict["field1"]
        _component = self.mediator.container._dict[recipient]
        _component.command = data_types.Command(request.command)
        _component.state.handle_command()

        self.mediator.transition_to(Command_Dispatched())
        self.mediator._state.return_to_idle()


class Command_Dispatched(State):
    def return_to_idle(self) -> None:
        self.mediator.transition_to(Idle())
        return
