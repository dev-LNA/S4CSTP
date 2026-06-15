import sys
from os.path import join
from threading import Thread

from PyQt6 import uic
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow

import func_tests.data_types as data_types
import func_tests.framework as framework
import func_tests.gui as gui


class GUI(QMainWindow):
    _START_BTN_CLICKED = False

    def __init__(self) -> None:
        super().__init__()

        # Load the UI page
        file_path = join("func_tests", "gui", "gui.ui")
        uic.loadUi(file_path, self)  # type: ignore
        self.setWindowTitle("S4ACS Functionalities Tests Framework")
        self.resize(400, 500)

        self.gui_widgets = gui.GUI_Widgets(self)
        self.gui_widgets.framework_stop_btn.clicked.connect(self.stop_application)
        self.gui_widgets.framework_start_btn.clicked.connect(self.start_application)
        self.gui_widgets.framework_run_tests_btn.clicked.connect(self.start_tests)

        s4acs = data_types.Component_Creator().create("fake")
        tests_list = data_types.Tests_List_Creator().create("fake")
        self.framework = framework.Functionalities_Tests_Framework(s4acs, tests_list)
        self._thread = Thread(target=self.framework.run)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_app)
        self.update_timer.start(200)

    def update_app(self) -> None:
        self._update_framework()
        self._update_comm_status()
        self._update_tests()
        self._update_s4acs()

    def _update_framework(self) -> None:
        # self.gui_widgets.framework_current_state.setText(self.framework.state)
        ...

    def _update_gui_obj(self, obj) -> None:
        obj.style().unpolish(obj)
        obj.style().polish(obj)
        obj.update()

    def _update_s4acs(self) -> None:
        s4acs_status = self.framework.s4acs.status
        if s4acs_status is None:
            return
        cam_status = s4acs_status["cam_status"]
        acq_config = s4acs_status["acq_config"]
        gui_widgets = self.gui_widgets

        led_status = {True: "on", False: "off"}

        gui_widgets.s4acs_led_power.setProperty(
            "led_status", led_status[cam_status["power"]]
        )
        self._update_gui_obj(gui_widgets.s4acs_led_power)
        gui_widgets.s4acs_led_exposing.setProperty(
            "led_status", led_status[cam_status["acquiring"]]
        )
        self._update_gui_obj(gui_widgets.s4acs_led_exposing)
        gui_widgets.s4acs_serial_number.setValue(cam_status["serial_number"])
        gui_widgets.s4acs_status.setText(cam_status["status"])
        gui_widgets.s4acs_last_image.setText(cam_status["last_image_name"])
        gui_widgets.s4acs_temperature.setValue(cam_status["current_temp"])
        gui_widgets.s4acs_exptime.setValue(cam_status["current_exp_time"])
        gui_widgets.s4acs_frames.setValue(acq_config["frames"])
        gui_widgets.s4acs_frames_done.setValue(cam_status["frames_done"])
        gui_widgets.s4acs_cycles.setValue(acq_config["cycles"])
        gui_widgets.s4acs_cycles_done.setValue(cam_status["cycles_done"])
        gui_widgets.s4acs_exe_status.setText(self.framework.s4acs.exe_status)

    def _update_tests(self) -> None:
        for idx, _test in enumerate(self.framework.tests_list):
            led, test_code, test_msg = self.gui_widgets.tests_dict[idx + 1]
            result = _test.result
            if result is None:
                continue
            led.setProperty("led_status", result.success)
            self._update_gui_obj(led)
            test_code.setText(result.test_code)
            test_msg.setText(result.message)

    def _update_comm_status(self) -> None:
        led_status = {True: "on", False: "off"}
        comm_status = self.framework.s4acs.return_comm_status()
        self.gui_widgets.s4acs_led_comm.setProperty(
            "led_status", led_status[comm_status]
        )
        self._update_gui_obj(self.gui_widgets.s4acs_led_comm)

    def stop_application(self) -> None:
        if self._START_BTN_CLICKED:
            self.framework.stop_thread = True
            self._thread.join()
            self.framework.end()
        sys.exit()

    def start_application(self) -> None:
        self._START_BTN_CLICKED = True
        self.gui_widgets.framework_start_btn.setDisabled(True)
        log_level = self._return_log_level()
        self.framework.log_level = log_level
        self.framework.initialize()
        self._thread.start()

    def _return_log_level(self) -> data_types.Log_Level:
        return {
            "DEBUG": data_types.Log_Level.DEBUG,
            "INFO": data_types.Log_Level.INFO,
            "WARNING": data_types.Log_Level.WARNING,
            "ERROR": data_types.Log_Level.ERROR,
            "CRITICAL": data_types.Log_Level.CRITICAL,
        }[self.gui_widgets.framework_log_level.currentText()]

    def start_tests(self) -> None:
        self.framework.start_tests = True
