import json

import func_tests.comm_channel as comm_channel


class External_Application:
    def __init__(self, _publisher: comm_channel.ZeroMQ_PUB) -> None:
        self.status: dict
        self._publisher = _publisher

    def publish_status(self) -> None:
        self._publisher.send_msg(json.dumps(self.status))
        return

    def initialize(self) -> None:
        self._publisher.initialize_comm()

    def end(self) -> None:
        self._publisher.close_comm()
