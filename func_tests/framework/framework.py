import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import astropy.io.fits as fits
import numpy as np

import func_tests.component as component
import func_tests.data_types as data_types
import func_tests.gui as gui
import func_tests.utils as utils


class Functionalities_Tests_Framework:
    stop_thread = False
    start_tests = False
    stop_tests = False
    stop_1st_err: bool

    def __init__(
        self,
        s4acs: component.S4ACS,
        _gui: gui.GUI,
    ) -> None:
        self.s4acs = s4acs
        self._external_apps = data_types.Framework_setup().create_external_apps()
        self._do_not_pub: list[str] = []
        self._gui = _gui
        self.log_dir = Path("func_tests/_logs")
        self.log_level = data_types.Log_Level.INFO
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_imgs_folder()
        return

    def set_tests_list(self, test_type: str, test_code: str = "") -> None:
        self.tests_list = data_types.Tests_List_Creator().create(test_type, test_code)

        return

    def initialize(self) -> None:
        self.create_log_file()
        logging.info("Framework was started")
        utils.run_s4acs_exe()
        self._initialize_s4acs()
        self._initialize_external_apps()
        for _test in self.tests_list:
            _test.s4acs = self.s4acs
            _test.framework = self
        logging.debug("Framework was initialized succesfully")
        return

    def _initialize_external_apps(self) -> None:
        for key, app in self._external_apps.items():
            app.initialize()
            logging.debug(f"External application {key.upper()} was started")

    def create_log_file(self) -> None:
        now = datetime.now(timezone.utc)
        log_file = self.log_dir / f"{now.strftime('%Y%m%d')}.log"
        datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")
        self._create_log_file_header(log_file, datetime_str)
        logging.basicConfig(
            filename=log_file,
            level=self.log_level,
            format="%(asctime)s [%(levelname)-8s] --> %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            filemode="a",
            force=True,
        )

    def _create_log_file_header(self, log_file: Path, datetime_str: str) -> None:
        with open(log_file, "w") as file:
            file.write(
                "\n========================================================================================\n"
                "S4ACS Functionalities Tests Framework - Event Log\n"
                "========================================================================================\n\n"
                "Description     : this file logs important events occured during the framework execution\n"
                "Version         : v0.1.0\n"
                "Log Type        : Operational Events\n"
                f"Log level       : {self.log_level.name}\n"
                f"Created at (UTC): {datetime_str}\n"
                "----------------------------------------------------------------------------------------\n"
                "Timestamp           Level          Message\n"
                "----------------------------------------------------------------------------------------\n\n"
            )
        return

    def _prepare_imgs_folder(self) -> None:
        logging.debug("Preparing images folder")
        cfg_file_content = utils.read_config_file()
        channel = cfg_file_content.channel
        imgs_folder = cfg_file_content.image_path
        destination_folder = imgs_folder / Path("../_temp")
        destination_folder.mkdir(exist_ok=True)
        imgs_list = os.listdir(imgs_folder)
        for img in imgs_list:
            Path(imgs_folder / img).replace(destination_folder / img)
        new_image = np.zeros((1024, 1024))
        new_image_name = imgs_folder / f"00000000_s4c{channel}_000010.fits"
        fits.writeto(new_image_name, new_image, overwrite=True)

    def _initialize_s4acs(self) -> None:
        self.s4acs.initialize()
        logging.debug("Setting default configuration...")
        self.s4acs.set_cam_config(utils.default_cam_config.copy())
        self.s4acs.set_acquisition_config(utils.default_acq_config.copy())

    # =============================================

    def end(self) -> None:
        if self.s4acs.return_comm_status():
            self.s4acs.send_command("STOP_APP")
            sleep(0.5)
        self.s4acs.end()
        for key, app in self._external_apps.items():
            app.end()
            logging.debug(f"External application {key.upper()} was stopped")
        logging.info("Framework was stopped")

    def update_status(self) -> None:
        while not self.stop_thread:
            self.s4acs.get_status_message()
            for key, app in self._external_apps.items():
                if key in self._do_not_pub:
                    continue
                app.publish_status()

    def run(self) -> None:
        while not self.stop_thread:
            sleep(0.025)
            if self.start_tests is True:
                self.clear_results()
                self.run_tests()
                logging.info("The tests were finished")
                self._gui.gui_widgets.framework_run_tests_btn.setEnabled(True)
                self._gui.gui_widgets.framework_stop_tests_btn.setDisabled(True)
                self.start_tests = False

    def run_tests(self) -> None:
        logging.info("Starting the tests...")
        for _test in self.tests_list:
            _test.s4acs.send_command(
                f"====== This is the test {_test._test_code} ======"
            )
            _test.set_result("warn", "")
            _test.run_test()

            if self.stop_1st_err and (_test.result.success == "error"):
                logging.info("An error was found. Stopping the tests...")
                return
            if self.stop_tests:
                logging.info("The tests were stopped")
                self.stop_tests = False
                return

    def clear_results(self) -> None:
        for _test in self.tests_list:
            _test.result = data_types.Test_Result(
                success="off", test_code="", message=""
            )
