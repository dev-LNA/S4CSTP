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

import func_tests.gui as gui


class GUI_Widgets:
    def __init__(self, window: gui.GUI) -> None:

        # ================ Master ================
        self.framework_current_state = window.findChild(
            QLineEdit, "framework_current_state"
        )
        self.stop_btn = window.findChild(QPushButton, "framework_stop_btn")
        self.start_btn = window.findChild(QPushButton, "framework_start_btn")

        # ================ S4ACS ================
        self.s4acs_led_comm = window.findChild(QLabel, "s4acs_led_comm")
        # self.s4acs_led_power = window.findChild(QLabel, "s4acs_led_power")
        # self.s4acs_led_exposing = window.findChild(QLabel, "s4acs_led_exposing")
        # self.s4acs_serial_number = window.findChild(QSpinBox, "s4acs_serial_number")
        # self.s4acs_status = window.findChild(QLineEdit, "s4acs_status")
        # self.s4acs_last_image = window.findChild(QLineEdit, "s4acs_last_image")
        # self.s4acs_temperature = window.findChild(QDoubleSpinBox, "s4acs_temperature")
        # self.s4acs_exptime = window.findChild(QDoubleSpinBox, "s4acs_exptime")
        # self.s4acs_frames = window.findChild(QSpinBox, "s4acs_frames")
        # self.s4acs_frames_done = window.findChild(QSpinBox, "s4acs_frames_done")
        # self.s4acs_cycles = window.findChild(QSpinBox, "s4acs_cycles")
        # self.s4acs_cycles_done = window.findChild(QSpinBox, "s4acs_cycles_done")
        # self.s4acs_exe_status = window.findChild(QLineEdit, "s4acs_exe_status")
