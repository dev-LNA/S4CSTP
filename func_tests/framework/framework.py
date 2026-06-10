import logging
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import func_tests.component as component
import func_tests.data_types as data_types


class Functionalities_Tests_Framework:
    _stop_thread = False

    def __init__(
        self,
        s4acs: component.Component,
    ) -> None:
        self.s4acs = s4acs
        self.log_dir = Path("func_tests/_logs")
        self.log_level = data_types.Log_Level.INFO

        self.log_dir.mkdir(parents=True, exist_ok=True)
        return

    def set_log_level(self, log_level: data_types.Log_Level) -> None:
        self.log_level = log_level
        return

    # ================ Execution ====================

    def initialize(self) -> None:
        self._state.initialize()

    def create_log_file(self) -> None:
        log_file = self.log_dir / "log.log"
        self._create_log_file_header(log_file)
        logging.basicConfig(
            filename=log_file,
            level=self.log_level,
            format="%(asctime)s [%(levelname)-8s] --> %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    def _create_log_file_header(self, log_file: Path) -> None:
        if log_file.exists():
            return
        with open(log_file, "a") as file:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            file.write(
                "\n========================================================================================\n"
                "S4ACS Functionalities Tests Framework - Event Log\n"
                "========================================================================================\n\n"
                "Description     : this file logs important events occured during the framework execution\n"
                "Version         : v0.1.0\n"
                "Log Type        : Operational Events\n"
                f"Log level       : {self.log_level.name}\n"
                f"Created at (UTC): {now}\n"
                "----------------------------------------------------------------------------------------\n"
                "Timestamp           Level          Message\n"
                "----------------------------------------------------------------------------------------\n\n"
            )
        return

    def end(self) -> None:
        self.s4acs.end()
        logging.info("Framework was stopped")

    def run(self) -> None:
        while not self._stop_thread:
            self.s4acs.get_status_message()
            sleep(0.2)

    # ================ Returns ====================

    def stop_thread(self) -> None:
        self._stop_thread = True
