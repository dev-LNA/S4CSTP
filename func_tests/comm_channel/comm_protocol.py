from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import zmq

import func_tests.data_types as data_types


class Communication_Channel(ABC):
    TIME_OUT = 50  # ms
    INTERVAL_BTW_MSGS = 5  # s

    def __init__(self, end_point: data_types.End_Point) -> None:
        super().__init__()
        self._end_point = end_point
        self._received_msg: str
        self.new_msg = False
        self._comm_status: bool = False
        self.last_msg_timestamp = datetime.now(timezone.utc) - timedelta(seconds=5)

    @property
    def received_msg(self) -> str:
        return self._received_msg

    @property
    def comm_status(self) -> bool:
        return self._comm_status

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
        delay = current_time_stemp - self.last_msg_timestamp
        self._comm_status = delay.seconds < self.INTERVAL_BTW_MSGS


class ZeroMQ_SUB(Communication_Channel):
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
        self.new_msg = self.socket.poll(timeout=self.TIME_OUT) != 0
        if self.new_msg:
            self._received_msg = self.socket.recv(zmq.NOBLOCK)
            self.last_msg_timestamp = datetime.now(timezone.utc)
        self._verify_comm_status()


class ZeroMQ_PUB(Communication_Channel):
    def __init__(
        self,
        end_point: data_types.End_Point,
        context: zmq.Context,
    ) -> None:
        super().__init__(end_point)
        self.context = context

    def initialize_comm(self) -> None:
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(self._end_point.to_str())

    def send_msg(self, msg: str) -> None:
        self.socket.send_string(msg)

    def close_comm(self) -> None:
        self.socket.close()


class ZeroMQ_REQ(Communication_Channel):
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
        self.new_msg = self.socket.poll(timeout=self.TIME_OUT) != 0
        if self.new_msg:
            self._received_msg = self.socket.recv()
            self.last_msg_timestamp = datetime.now(timezone.utc)


class Fake_Subscriber(Communication_Channel):
    def initialize_comm(self) -> None: ...

    def close_comm(self) -> None: ...

    def receive_msg(self) -> None:
        self.last_msg_timestamp = datetime.now(timezone.utc)
        self._verify_comm_status()


class Fake_Requester(Communication_Channel):
    def initialize_comm(self) -> None: ...

    def close_comm(self) -> None: ...

    def receive_msg(self) -> None:
        self._received_msg = "ACK"
        self.last_msg_timestamp = datetime.now(timezone.utc)

    def send_msg(self, msg: str) -> None: ...
