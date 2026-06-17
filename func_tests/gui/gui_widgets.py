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
        self.framework_current_status = window.findChild(
            QLineEdit, "framework_current_status"
        )
        self.framework_log_level = window.findChild(QComboBox, "framework_log_level")
        self.framework_stop_1st_err = window.findChild(
            QPushButton, "framework_stop_1st_err"
        )
        self.framework_stop_btn = window.findChild(QPushButton, "framework_stop_btn")
        self.framework_start_btn = window.findChild(QPushButton, "framework_start_btn")
        self.framework_run_tests_btn = window.findChild(
            QPushButton, "framework_run_tests_btn"
        )

        # ================ S4ACS ================
        self.s4acs_led_comm = window.findChild(QLabel, "s4acs_led_comm")
        self.s4acs_led_power = window.findChild(QLabel, "s4acs_led_power")
        self.s4acs_led_exposing = window.findChild(QLabel, "s4acs_led_exposing")
        self.s4acs_serial_number = window.findChild(QSpinBox, "s4acs_serial_number")
        self.s4acs_status = window.findChild(QLineEdit, "s4acs_status")
        self.s4acs_last_image = window.findChild(QLineEdit, "s4acs_last_image")
        self.s4acs_temperature = window.findChild(QDoubleSpinBox, "s4acs_temperature")
        self.s4acs_exptime = window.findChild(QDoubleSpinBox, "s4acs_exptime")
        self.s4acs_frames = window.findChild(QSpinBox, "s4acs_frames")
        self.s4acs_frames_done = window.findChild(QSpinBox, "s4acs_frames_done")
        self.s4acs_cycles = window.findChild(QSpinBox, "s4acs_cycles")
        self.s4acs_cycles_done = window.findChild(QSpinBox, "s4acs_cycles_done")
        self.s4acs_exe_status = window.findChild(QLineEdit, "s4acs_exe_status")

        # ================ Tests ================
        self.tests_dict = {
            1: [
                window.findChild(QLabel, "tests_led1"),
                window.findChild(QLineEdit, "tests_test_code1"),
                window.findChild(QLineEdit, "tests_message1"),
            ],
            2: [
                window.findChild(QLabel, "tests_led2"),
                window.findChild(QLineEdit, "tests_test_code2"),
                window.findChild(QLineEdit, "tests_message2"),
            ],
            3: [
                window.findChild(QLabel, "tests_led3"),
                window.findChild(QLineEdit, "tests_test_code3"),
                window.findChild(QLineEdit, "tests_message3"),
            ],
            4: [
                window.findChild(QLabel, "tests_led4"),
                window.findChild(QLineEdit, "tests_test_code4"),
                window.findChild(QLineEdit, "tests_message4"),
            ],
            5: [
                window.findChild(QLabel, "tests_led5"),
                window.findChild(QLineEdit, "tests_test_code5"),
                window.findChild(QLineEdit, "tests_message5"),
            ],
            6: [
                window.findChild(QLabel, "tests_led6"),
                window.findChild(QLineEdit, "tests_test_code6"),
                window.findChild(QLineEdit, "tests_message6"),
            ],
            7: [
                window.findChild(QLabel, "tests_led7"),
                window.findChild(QLineEdit, "tests_test_code7"),
                window.findChild(QLineEdit, "tests_message7"),
            ],
            8: [
                window.findChild(QLabel, "tests_led8"),
                window.findChild(QLineEdit, "tests_test_code8"),
                window.findChild(QLineEdit, "tests_message8"),
            ],
            9: [
                window.findChild(QLabel, "tests_led9"),
                window.findChild(QLineEdit, "tests_test_code9"),
                window.findChild(QLineEdit, "tests_message9"),
            ],
            10: [
                window.findChild(QLabel, "tests_led10"),
                window.findChild(QLineEdit, "tests_test_code10"),
                window.findChild(QLineEdit, "tests_message10"),
            ],
            11: [
                window.findChild(QLabel, "tests_led11"),
                window.findChild(QLineEdit, "tests_test_code11"),
                window.findChild(QLineEdit, "tests_message11"),
            ],
            12: [
                window.findChild(QLabel, "tests_led12"),
                window.findChild(QLineEdit, "tests_test_code12"),
                window.findChild(QLineEdit, "tests_message12"),
            ],
            13: [
                window.findChild(QLabel, "tests_led13"),
                window.findChild(QLineEdit, "tests_test_code13"),
                window.findChild(QLineEdit, "tests_message13"),
            ],
            14: [
                window.findChild(QLabel, "tests_led14"),
                window.findChild(QLineEdit, "tests_test_code14"),
                window.findChild(QLineEdit, "tests_message14"),
            ],
            15: [
                window.findChild(QLabel, "tests_led15"),
                window.findChild(QLineEdit, "tests_test_code15"),
                window.findChild(QLineEdit, "tests_message15"),
            ],
            16: [
                window.findChild(QLabel, "tests_led16"),
                window.findChild(QLineEdit, "tests_test_code16"),
                window.findChild(QLineEdit, "tests_message16"),
            ],
            17: [
                window.findChild(QLabel, "tests_led17"),
                window.findChild(QLineEdit, "tests_test_code17"),
                window.findChild(QLineEdit, "tests_message17"),
            ],
            18: [
                window.findChild(QLabel, "tests_led18"),
                window.findChild(QLineEdit, "tests_test_code18"),
                window.findChild(QLineEdit, "tests_message18"),
            ],
            19: [
                window.findChild(QLabel, "tests_led19"),
                window.findChild(QLineEdit, "tests_test_code19"),
                window.findChild(QLineEdit, "tests_message19"),
            ],
            20: [
                window.findChild(QLabel, "tests_led20"),
                window.findChild(QLineEdit, "tests_test_code20"),
                window.findChild(QLineEdit, "tests_message20"),
            ],
            21: [
                window.findChild(QLabel, "tests_led21"),
                window.findChild(QLineEdit, "tests_test_code21"),
                window.findChild(QLineEdit, "tests_message21"),
            ],
            22: [
                window.findChild(QLabel, "tests_led22"),
                window.findChild(QLineEdit, "tests_test_code22"),
                window.findChild(QLineEdit, "tests_message22"),
            ],
            23: [
                window.findChild(QLabel, "tests_led23"),
                window.findChild(QLineEdit, "tests_test_code23"),
                window.findChild(QLineEdit, "tests_message23"),
            ],
            24: [
                window.findChild(QLabel, "tests_led24"),
                window.findChild(QLineEdit, "tests_test_code24"),
                window.findChild(QLineEdit, "tests_message24"),
            ],
            25: [
                window.findChild(QLabel, "tests_led25"),
                window.findChild(QLineEdit, "tests_test_code25"),
                window.findChild(QLineEdit, "tests_message25"),
            ],
            26: [
                window.findChild(QLabel, "tests_led26"),
                window.findChild(QLineEdit, "tests_test_code26"),
                window.findChild(QLineEdit, "tests_message26"),
            ],
            27: [
                window.findChild(QLabel, "tests_led27"),
                window.findChild(QLineEdit, "tests_test_code27"),
                window.findChild(QLineEdit, "tests_message27"),
            ],
        }
