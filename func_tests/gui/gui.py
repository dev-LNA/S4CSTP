import sys
from os.path import join
from threading import Thread

from PyQt6 import uic
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow

import func_tests.data_types as data_types
import func_tests.ftf as ftf
import func_tests.gui as gui


class GUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        # Load the UI page
        file_path = join("func_tests", "gui", "gui.ui")
        uic.loadUi(file_path, self)  # type: ignore
        self.setWindowTitle("S4ACS Functionalities Tests Framework")
        # self.resize(1150, 400)

        self.gui_widgets = gui.GUI_Widgets(self)
        self.gui_widgets.stop_btn.clicked.connect(self.stop_application)
        self.gui_widgets.start_btn.clicked.connect(self.start_application)

        s4acs = data_types.Component_Creator().create("fake")
        self.ftf = ftf.Functionalities_Tests_Framework(s4acs)
        # self._thread = Thread(target=self.ftf.run)

        # self.update_timer = QTimer(self)
        # self.update_timer.timeout.connect(self.update_app)
        # self.update_timer.start(200)

    def update_app(self) -> None:
        instrument_status = self.ftf.return_components_status()
        self._update_master()
        self._update_s4acs(instrument_status)

    def _update_master(self) -> None:
        self.gui_widgets.ftf_current_state.setText(self.ftf.state)

    def _update_gui_obj(self, obj) -> None:
        obj.style().unpolish(obj)
        obj.style().polish(obj)
        obj.update()

    def _update_s4acs(self, instrument_status) -> None:
        acs_status = instrument_status["EACS1"]
        if acs_status is None:
            return
        cam_status = acs_status["cam_status"]
        acq_config = acs_status["acq_config"]
        gui_widgets = self.gui_widgets

        led_status = {True: "on", False: "off"}

        gui_widgets.eacs1_led_power.setProperty(
            "led_status", led_status[cam_status["power"]]
        )
        self._update_gui_obj(gui_widgets.eacs1_led_power)
        gui_widgets.eacs1_led_exposing.setProperty(
            "led_status", led_status[cam_status["acquiring"]]
        )
        self._update_gui_obj(gui_widgets.eacs1_led_exposing)
        gui_widgets.eacs1_serial_number.setValue(cam_status["serial_number"])
        gui_widgets.eacs1_status.setText(cam_status["status"])
        gui_widgets.eacs1_last_image.setText(cam_status["last_image_name"])
        gui_widgets.eacs1_temperature.setValue(cam_status["current_temp"])
        gui_widgets.eacs1_exptime.setValue(cam_status["current_exp_time"])
        gui_widgets.eacs1_frames.setValue(acq_config["frames"])
        gui_widgets.eacs1_frames_done.setValue(cam_status["frames_done"])
        gui_widgets.eacs1_cycles.setValue(acq_config["cycles"])
        gui_widgets.eacs1_cycles_done.setValue(cam_status["cycles_done"])
        gui_widgets.eacs1_exe_status.setText(self.ftf.container.eacs1.exe_status)

    def _update_comm_status(self) -> None:
        led_status = {True: "on", False: "off"}
        comm_status = self.ftf.return_comm_status()
        self.gui_widgets.eacs1_led_comm.setProperty(
            "led_status", led_status[comm_status["EACS1"]]
        )
        self._update_gui_obj(self.gui_widgets.eacs1_led_comm)

    def stop_application(self) -> None:
        # self.ftf.stop_thread()
        # self._thread.join()
        self.ftf.end()
        sys.exit()

    def start_application(self) -> None:
        self.ftf.initialize()
        # self._thread.start()
