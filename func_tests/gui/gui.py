import sys
from os.path import join
from threading import Thread

from PyQt6 import uic
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow

import src.gui as gui
import src.state.mediator as med_state
from src.data_types import Log_Level, Mediator_Setup
from src.mediator import EMCS


class EMCS_GUI(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        # Load the UI page
        file_path = join("src", "gui", "gui.ui")
        uic.loadUi(file_path, self)  # type: ignore
        self.setWindowTitle("ECHARPE Master Control System")
        self.resize(1150, 400)

        self.gui_widgets = gui.GUI_Widgets(self)
        self.gui_widgets.stop_btn.clicked.connect(self.stop_application)
        self.gui_widgets.start_btn.clicked.connect(self.start_application)

        _client, container = Mediator_Setup().create("test_acs1_client")
        self.emcs = EMCS(container, _client, med_state.Not_Initialized())
        self._thread = Thread(target=self.emcs.run)

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_app)
        self.update_timer.start(200)

    def _return_log_level(self) -> Log_Level:
        return {
            "DEBUG": Log_Level.DEBUG,
            "INFO": Log_Level.INFO,
            "WARNING": Log_Level.WARNING,
            "ERROR": Log_Level.ERROR,
            "CRITICAL": Log_Level.CRITICAL,
        }[self.gui_widgets.emcs_log_level.currentText()]

    def update_app(self) -> None:
        instrument_status = self.emcs.return_components_status()
        self._update_master()
        self._update_eacs1(instrument_status)
        self._update_eacs2(instrument_status)
        self._update_focuser(instrument_status)
        self._update_client()
        self._update_comm_status()

    def _update_master(self) -> None:
        self.gui_widgets.emcs_current_state.setText(self.emcs.state)

    def _update_client(self) -> None:
        request = self.emcs.client.request
        gui_widgets = self.gui_widgets
        if request is not None:
            _component = self.emcs.container._dict[request._dict["field1"]]
            command = _component.command

            self.gui_widgets.client_request.setText(request.str)
            time_stamp = self.emcs.client.time_stamp
            gui_widgets.client_request_timestamp.setSpecialValueText(time_stamp)
            gui_widgets.client_led_request.setProperty("led_status", request.status)
            self._update_gui_obj(gui_widgets.client_led_request)
            gui_widgets.client_led_recipient.setProperty(
                "led_status", request.recipient
            )
            self._update_gui_obj(gui_widgets.client_led_recipient)
            gui_widgets.client_led_command.setProperty("led_status", command.supported)
            self._update_gui_obj(gui_widgets.client_led_command)
            gui_widgets.client_led_timeout.setProperty("led_status", command.timeout)
            self._update_gui_obj(gui_widgets.client_led_timeout)

    def _update_gui_obj(self, obj) -> None:
        obj.style().unpolish(obj)
        obj.style().polish(obj)
        obj.update()

    def _update_eacs1(self, instrument_status) -> None:
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
        gui_widgets.eacs1_exe_status.setText(self.emcs.container.eacs1.exe_status)

    def _update_eacs2(self, instrument_status) -> None:
        acs_status = instrument_status["EACS2"]
        if acs_status is None:
            return
        cam_status = acs_status["cam_status"]
        acq_config = acs_status["acq_config"]
        gui_widgets = self.gui_widgets
        led_status = {True: "on", False: "off"}

        gui_widgets.eacs2_led_power.setProperty(
            "led_status", led_status[cam_status["power"]]
        )
        self._update_gui_obj(gui_widgets.eacs2_led_power)
        gui_widgets.eacs2_led_exposing.setProperty(
            "led_status", led_status[cam_status["acquiring"]]
        )
        self._update_gui_obj(gui_widgets.eacs2_led_exposing)
        gui_widgets.eacs2_serial_number.setValue(cam_status["serial_number"])
        gui_widgets.eacs2_status.setText(cam_status["status"])
        gui_widgets.eacs2_last_image.setText(cam_status["last_image_name"])
        gui_widgets.eacs2_temperature.setValue(cam_status["current_temp"])
        gui_widgets.eacs2_exptime.setValue(cam_status["current_exp_time"])
        gui_widgets.eacs2_frames.setValue(acq_config["frames"])
        gui_widgets.eacs2_frames_done.setValue(cam_status["frames_done"])
        gui_widgets.eacs2_cycles.setValue(acq_config["cycles"])
        gui_widgets.eacs2_cycles_done.setValue(cam_status["cycles_done"])

    def _update_focuser(self, instrument_status) -> None:
        focuser_status = instrument_status["FOCUSER"]
        if focuser_status is None:
            return
        led_status = {True: "on", False: "off"}
        self.gui_widgets.focuser_home.setProperty(
            "led_status", led_status[focuser_status["home"]]
        )
        self._update_gui_obj(self.gui_widgets.focuser_home)
        self.gui_widgets.focuser_moving.setProperty(
            "led_status", led_status[focuser_status["moving"]]
        )
        self._update_gui_obj(self.gui_widgets.focuser_moving)
        self.gui_widgets.focuser_position.setProperty(
            "led_status", led_status[focuser_status["position"]]
        )
        self._update_gui_obj(self.gui_widgets.focuser_position)

    def _update_comm_status(self) -> None:
        led_status = {True: "on", False: "off"}
        comm_status = self.emcs.return_comm_status()
        self.gui_widgets.eacs1_led_comm.setProperty(
            "led_status", led_status[comm_status["EACS1"]]
        )
        self._update_gui_obj(self.gui_widgets.eacs1_led_comm)
        self.gui_widgets.eacs2_led_comm.setProperty(
            "led_status", led_status[comm_status["EACS2"]]
        )
        self._update_gui_obj(self.gui_widgets.eacs2_led_comm)
        self.gui_widgets.focuser_comm.setProperty(
            "led_status", led_status[comm_status["FOCUSER"]]
        )
        self._update_gui_obj(self.gui_widgets.focuser_comm)

    def stop_application(self) -> None:
        self.emcs.stop_thread()
        self._thread.join()
        self.emcs.end()
        sys.exit()

    def start_application(self) -> None:
        log_level = self._return_log_level()
        self.emcs.set_log_level(log_level)
        self.emcs.initialize()
        self._thread.start()
