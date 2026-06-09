from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDateTimeEdit,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
)

import src.gui as gui


class GUI_Widgets:
    def __init__(self, window: gui.EMCS_GUI) -> None:

        # ================ Master ================
        self.emcs_current_state = window.findChild(QLineEdit, "emcs_current_state")
        self.emcs_log_level = window.findChild(QComboBox, "emcs_log_level")

        # self.emcs_log_level.setEditable(True)
        # qline_edit = self.emcs_log_level.lineEdit()
        # if qline_edit is None:
        #     raise RuntimeError("QLineEdit = None!")
        # qline_edit.setReadOnly(True)
        # qline_edit.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.stop_btn = window.findChild(QPushButton, "emcs_stop_btn")
        self.start_btn = window.findChild(QPushButton, "emcs_start_btn")

        # ================ EACS1 ================
        self.eacs1_led_comm = window.findChild(QLabel, "eacs1_led_comm")
        self.eacs1_led_power = window.findChild(QLabel, "eacs1_led_power")
        self.eacs1_led_exposing = window.findChild(QLabel, "eacs1_led_exposing")
        self.eacs1_serial_number = window.findChild(QSpinBox, "eacs1_serial_number")
        self.eacs1_status = window.findChild(QLineEdit, "eacs1_status")
        self.eacs1_last_image = window.findChild(QLineEdit, "eacs1_last_image")
        self.eacs1_temperature = window.findChild(QDoubleSpinBox, "eacs1_temperature")
        self.eacs1_exptime = window.findChild(QDoubleSpinBox, "eacs1_exptime")
        self.eacs1_frames = window.findChild(QSpinBox, "eacs1_frames")
        self.eacs1_frames_done = window.findChild(QSpinBox, "eacs1_frames_done")
        self.eacs1_cycles = window.findChild(QSpinBox, "eacs1_cycles")
        self.eacs1_cycles_done = window.findChild(QSpinBox, "eacs1_cycles_done")
        self.eacs1_exe_status = window.findChild(QLineEdit, "eacs1_exe_status")

        # ================ EACS2 ================
        self.eacs2_led_comm = window.findChild(QLabel, "eacs2_led_comm")
        self.eacs2_led_power = window.findChild(QLabel, "eacs2_led_power")
        self.eacs2_led_exposing = window.findChild(QLabel, "eacs2_led_exposing")
        self.eacs2_serial_number = window.findChild(QSpinBox, "eacs2_serial_number")
        self.eacs2_status = window.findChild(QLineEdit, "eacs2_status")
        self.eacs2_last_image = window.findChild(QLineEdit, "eacs2_last_image")
        self.eacs2_temperature = window.findChild(QDoubleSpinBox, "eacs2_temperature")
        self.eacs2_exptime = window.findChild(QDoubleSpinBox, "eacs2_exptime")
        self.eacs2_frames = window.findChild(QSpinBox, "eacs2_frames")
        self.eacs2_frames_done = window.findChild(QSpinBox, "eacs2_frames_done")
        self.eacs2_cycles = window.findChild(QSpinBox, "eacs2_cycles")
        self.eacs2_cycles_done = window.findChild(QSpinBox, "eacs2_cycles_done")
        self.eacs2_exe_status = window.findChild(QLineEdit, "eacs2_exe_status")

        # ================ Client ================
        self.client_request = window.findChild(QLineEdit, "client_request")
        self.client_request_timestamp = window.findChild(
            QDateTimeEdit, "client_request_timestamp"
        )
        self.client_led_request = window.findChild(QLabel, "client_led_request")
        self.client_led_recipient = window.findChild(QLabel, "client_led_recipient")
        self.client_led_command = window.findChild(QLabel, "client_led_command")
        self.client_led_timeout = window.findChild(QLabel, "client_led_timeout")
        # ================ Focuser ================
        self.focuser_comm = window.findChild(QLabel, "focuser_comm")
        self.focuser_home = window.findChild(QLabel, "focuser_home")
        self.focuser_moving = window.findChild(QLabel, "focuser_moving")
        self.focuser_position = window.findChild(QDoubleSpinBox, "focuser_position")
