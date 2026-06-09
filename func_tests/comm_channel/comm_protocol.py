from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import zmq

import src.data_types as data_types


class Communication_Protocol(ABC):
    TIME_OUT = 200  # ms
    INTERVAL_BTW_MSGS = 5  # s

    def __init__(self, end_point: data_types.End_Point) -> None:
        super().__init__()
        self._end_point = end_point
        self._received_msg: str
        self._new_msg = False
        self._comm_status: bool = False
        self._last_msg_timestamp = datetime.now(timezone.utc) - timedelta(seconds=5)

    @property
    def last_msg_timestamp(self) -> str:
        return self._last_msg_timestamp.strftime("%H:%M:%S")

    @property
    def received_msg(self) -> str:
        return self._received_msg

    @property
    def comm_status(self) -> bool:
        return self._comm_status

    @property
    def new_msg(self) -> bool:
        return self._new_msg

    @abstractmethod
    def initialize_comm(self) -> None: ...

    @abstractmethod
    def close_comm(self) -> None: ...

    def send_msg(self, msg: str) -> None:
        raise RuntimeError(
            f"The class {type(self).__name__} does not implement the send_msg method."
        )

    def receive_msg(self) -> None:
        raise RuntimeError(
            f"The class {type(self).__name__} does not implement the receive_msg method."
        )

    def _verify_comm_status(self) -> None:
        current_time_stemp = datetime.now(timezone.utc)
        delay = current_time_stemp - self._last_msg_timestamp
        self._comm_status = delay.seconds < self.INTERVAL_BTW_MSGS


class ZeroMQ_SUB(Communication_Protocol):
    def __init__(
        self,
        end_point: data_types.End_Point,
        context: zmq.Context,
    ) -> None:
        super().__init__(end_point)
        self.context = context

    def initialize_comm(self) -> None:
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(self._end_point.to_str())
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

    def close_comm(self) -> None:
        self.socket.close()

    def receive_msg(self) -> None:
        self._new_msg = self.socket.poll(timeout=self.TIME_OUT) != 0
        if self._new_msg:
            self._received_msg = self.socket.recv(zmq.NOBLOCK)
            self._last_msg_timestamp = datetime.now(timezone.utc)
        self._verify_comm_status()


class ZeroMQ_REQ(Communication_Protocol):
    def __init__(self, end_point: data_types.End_Point, context: zmq.Context) -> None:
        super().__init__(end_point)
        self.context = context

    def initialize_comm(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self._end_point.to_str())

    def close_comm(self) -> None:
        self.socket.close()

    def send_msg(self, msg: str) -> None:
        self.socket.send_string(msg)

    def receive_msg(self) -> None:
        self._new_msg = self.socket.poll(timeout=self.TIME_OUT) != 0
        if self._new_msg:
            self._received_msg = self.socket.recv()
            self._last_msg_timestamp = datetime.now(timezone.utc)


class ZeroMQ_REP(Communication_Protocol):
    def __init__(self, end_point: data_types.End_Point, context: zmq.Context) -> None:
        super().__init__(end_point)
        self.context = context

    def initialize_comm(self) -> None:
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(self._end_point.to_str())

    def close_comm(self) -> None:
        self.socket.close()

    def receive_msg(self) -> None:
        self._new_msg = self.socket.poll(timeout=self.TIME_OUT) != 0
        if self._new_msg:
            self._received_msg = self.socket.recv_string(zmq.NOBLOCK)
            self.socket.send_string("ACK")
            self._last_msg_timestamp = datetime.now(timezone.utc)
            return
        self._received_msg = ""


class MQTT(Communication_Protocol): ...


class Fake_Replier(Communication_Protocol):
    def initialize_comm(self) -> None: ...

    def close_comm(self) -> None: ...

    def receive_msg(self) -> None:
        self._last_msg_timestamp = datetime.now(timezone.utc)
        return


class Fake_Subscriber(Communication_Protocol):
    def initialize_comm(self) -> None: ...

    def close_comm(self) -> None: ...

    def receive_msg(self) -> None:
        self._last_msg_timestamp = datetime.now(timezone.utc)
        self._verify_comm_status()


class Fake_Requester(Communication_Protocol):
    def initialize_comm(self) -> None: ...

    def close_comm(self) -> None: ...

    def receive_msg(self) -> None:
        self._received_msg = "ACK"
        self._last_msg_timestamp = datetime.now(timezone.utc)

    def send_msg(self, msg: str) -> None: ...
