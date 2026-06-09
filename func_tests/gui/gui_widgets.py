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
        # self.ftf_current_state = window.findChild(QLineEdit, "ftf_current_state")

        self.stop_btn = window.findChild(QPushButton, "ftf_stop_btn")
        self.start_btn = window.findChild(QPushButton, "ftf_start_btn")

        # ================ EACS1 ================
        # self.eacs1_led_comm = window.findChild(QLabel, "eacs1_led_comm")
        # self.eacs1_led_power = window.findChild(QLabel, "eacs1_led_power")
        # self.eacs1_led_exposing = window.findChild(QLabel, "eacs1_led_exposing")
        # self.eacs1_serial_number = window.findChild(QSpinBox, "eacs1_serial_number")
        # self.eacs1_status = window.findChild(QLineEdit, "eacs1_status")
        # self.eacs1_last_image = window.findChild(QLineEdit, "eacs1_last_image")
        # self.eacs1_temperature = window.findChild(QDoubleSpinBox, "eacs1_temperature")
        # self.eacs1_exptime = window.findChild(QDoubleSpinBox, "eacs1_exptime")
        # self.eacs1_frames = window.findChild(QSpinBox, "eacs1_frames")
        # self.eacs1_frames_done = window.findChild(QSpinBox, "eacs1_frames_done")
        # self.eacs1_cycles = window.findChild(QSpinBox, "eacs1_cycles")
        # self.eacs1_cycles_done = window.findChild(QSpinBox, "eacs1_cycles_done")
        # self.eacs1_exe_status = window.findChild(QLineEdit, "eacs1_exe_status")
