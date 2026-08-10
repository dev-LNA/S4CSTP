import socket
from collections.abc import Sequence

import zmq

import func_tests.comm_channel as comm_channel
import func_tests.component as component
import func_tests.external_app as external_app
import func_tests.strategy as strategy
import func_tests.utils as utils
from func_tests.data_types import End_Point


class Framework_Setup:
    tests_list = [
        strategy.I001,
        strategy.I002,
        strategy.I003,
        strategy.I004,
        strategy.I005,
        strategy.I006,
        strategy.I007,
        strategy.E001,
        strategy.E002,
        strategy.E003,
        strategy.E005,
        strategy.E006,
        strategy.E007,
        strategy.E008,
        strategy.E009,
        strategy.E010,
        strategy.E011,
        strategy.E012,
        strategy.E014,
        strategy.E015,
        strategy.E019,
        strategy.S001,
    ]

    complex_tests = [
        strategy.I004,
        strategy.I006,
        strategy.I007,
        strategy.E002,
        strategy.E014,
        strategy.E015,
    ]

    def __init__(self) -> None:
        self.localhost: str = self.get_local_ip()

    def create_component(self, _type: str) -> component.S4ACS:
        """create a Component

        Args:
            _type (str): construtor type

        Returns:
            component.Component: _description_
        """
        if _type == "fake":
            end_point = End_Point(ip="192.168.0.1", port=5555)
            subscriber = comm_channel.Fake_Subscriber(end_point)
            requester = comm_channel.Fake_Requester(end_point)
            return component.Fake_Component(subscriber, requester)

        if _type == "real":
            context = zmq.Context()
            end_point = End_Point(ip=self.localhost, port=5555)
            subscriber = comm_channel.ZeroMQ_SUB(end_point, context)
            end_point = End_Point(ip=self.localhost, port=5556)
            requester = comm_channel.ZeroMQ_REQ(end_point, context)
            return component.S4ACS(subscriber, requester)

        else:
            raise ValueError(f"Unknown type: {_type}")

    def create_external_apps(self) -> dict[str, external_app.External_Application]:
        """Create external applications

        Returns:
            dict[str, external_app.External_Application]
        """

        socket_config = utils.read_config_file("socket.cfg")
        context = zmq.Context()

        end_point_str = socket_config.get("SUB_GUI", "address")
        end_point = End_Point.from_str(end_point_str)
        publisher = comm_channel.ZeroMQ_PUB(end_point, context)
        s4gui = external_app.External_Application(publisher)
        s4gui.status = utils.S4GUI_JSON.copy()

        end_point_str = socket_config.get("SUB_ICS", "address")
        end_point = End_Point.from_str(end_point_str)
        publisher = comm_channel.ZeroMQ_PUB(end_point, context)
        s4ics = external_app.External_Application(publisher)
        s4ics.status = utils.S4ICS_JSON

        end_point_str = socket_config.get("SUB_TCS", "address")
        end_point = End_Point.from_str(end_point_str)
        publisher = comm_channel.ZeroMQ_PUB(end_point, context)
        tcs = external_app.External_Application(publisher)
        tcs.status = utils.TCS_JSON.copy()

        end_point_str = socket_config.get("SUB_WSTATION", "address")
        end_point = End_Point.from_str(end_point_str)
        publisher = comm_channel.ZeroMQ_PUB(end_point, context)
        weather = external_app.External_Application(publisher)
        weather.status = utils.WEATHER_JSON.copy()

        end_point_str = socket_config.get("SUB_FOCUSER", "address")
        end_point = End_Point.from_str(end_point_str)
        publisher = comm_channel.ZeroMQ_PUB(end_point, context)
        focuser = external_app.External_Application(publisher)
        focuser.status = utils.FOCUSER_JSON.copy()
        return {
            "s4gui": s4gui,
            "s4ics": s4ics,
            "weather_st": weather,
            "tcs": tcs,
            "focuser": focuser,
        }

    def create_tests_list(
        self, _type: str, test_code: str = ""
    ) -> Sequence[strategy.Test_Strategy]:
        """Create list of tests

        Args:
            _type (str): test type. Allowed values are fake, all tests, quick tests, init tests, exe tests and one test
            test_code (str, optional): For a single test only. Defaults to "".

        Raises:
            ValueError: if the test code was not found
            ValueError: if the test type is unkonwn

        Returns:
            Sequence[strategy.Test_Strategy]: list of tests
        """
        if _type == "fake":
            return [strategy.Fake_Positive_Test() for _ in range(17)] + [
                strategy.Fake_Negative_Test() for _ in range(10)
            ]
        if _type == "all tests":
            return [_test() for _test in self.tests_list]
        if _type == "quick tests":
            return [
                _test() for _test in self.tests_list if _test not in self.complex_tests
            ]

        if _type == "init tests":
            return [_test() for _test in self.tests_list if "I" in _test.__name__]
        if _type == "exe tests":
            return [_test() for _test in self.tests_list if "E" in _test.__name__]
        if _type == "one test":
            for _test in self.tests_list:
                if _test.__name__ == test_code:
                    return [_test()]
            raise ValueError(f"Test does no found: {test_code}")
        if _type == "start with":
            return [
                _test()
                for _test in self.tests_list
                if int(_test.__name__[1:]) >= int(test_code[1:])
                and test_code[0] in _test.__name__
            ]
        raise ValueError(f"Unknown type: {_type}")

    def get_local_ip(self) -> str:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
