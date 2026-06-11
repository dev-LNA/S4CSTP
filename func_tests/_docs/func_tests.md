This page presents a list of tests desgined to validate each S4ACS functionalities. 
These tests should be run before releasing a new S4ACS version

# When running S4ACS
- [ ] **I001** - The configuration used to initiate S4ACS should be validated.
- [ ] **I002** - S4ACS should get the pu8blication of S4GUI, S4ICS, Focuser, Weather Station and TCS. 
- [ ] **I003** - Three log files should be created: the events log file, errors log file, and keywords log file.
- [ ] **I004** - S4ACS should validate the provided path for the images folder.
- [ ] **I005** - The index of the last acquired image inside the folder should be determined.
- [ ] **I006** - The camera should be initialized succesfully.

# During the S4ACS execution

- [ ] **E001** -  S4ACS should publish a status message every 1000 ms when in the idle state, and every 200 ms in any other case.
- [ ] **E002** -  A message should be logged in the events log file everytime S4ACS restablish/loose communication with an external application. In addition, the status of this communication should also be written in the published status message.
- [ ] **E003** -  When running, S4ACS should log several messages as a function of the log level STATUS, DEBBUG, INFO, WARNING, ERROR, and CRITICAL.
- [ ] **E004** -  S4ACS should create three new log files every day at 12h UTC.
----------------------------------------------------------------------------------------

- [ ] **E005** The commands `SET, STOP_APP, WRITE_SETUP, WAIT_EXPOSE_COMMAND` should be accepted only if S4ACS is in the IDLE state. Otherwise, a warning should be logged.
- [ ] **E006** When receiveing a `STOP_APP` command, S4ACS should stop execution.
- [ ] **E007** An `EXPOSE` command should be accepted only if there is no error in the provided operation mode or if S4ACS is in the `STOP_ACQUISITION`, `WAIT_EXPOSE_COMMAND` states. Otherwise, a warning should be logged.
- [ ] **E008** In receiving an `EXPOSE` command, S4ACS should validate the current condition of the retarder waveplate, in case of polarimetric acquisitions. 
- [ ] **E009** The `STOP_ACQUISITION` command should be accepted only before the end of an image series. Otherwise, a warning should be logged. If the acquisition is stopped, S4ACS should finish the acquisition of the current sequence/cycle and change to the IDLE state.
- [ ] **E009** The `PAUSE_ACQUISITION` command should be accepted only before the end of the acquisition of a photometric image series. Otherwise, a warning should be logged. If the acquisition is paused, S4ACS should wait for the resume command request.
- [ ] **E010** A `RESUME_ACQUISITION` command should be accepted only if the current acquisition is paused. Otherwise, a warning should be logged.
- [ ] **E011** An `ABORT_ACQUISITION` command should be accepted only during an exposition or before the end of an image series. In case no condition is met, a warning should be logged. Besides, when receiving an abort acquisition request, the current acquisition should be aborted and S4ACS should be set to the IDLE state.

----------------------------------------------------------------------------------------

- [ ] **E012** - When receiving a request for a new operation mode, the provided parameters should be validated as a function of the allowed intervals and predefined values, when applicable. Moreover, there are some parameters that present an interdependence, which are sutther and sub-image parameters. These parameters should be tested for valid and invalid values. In case of invalid values, a respective warning should be logged in the events log file.
- [ ] **E013** - The camera operation mode should be set only if no error is detected in the operation mode
- [ ] **E014** - When receiveing a request for configuring one fo the acquisition parameters, the provided value should be validated as a function of the allowed interval or predefined values, when applicable. Besides, an error should be logged in the case of any inconsistency. 
- [ ] **E015** - The request for configuring a new acquisition parameter should be accepted only if the provided value is valid.
- [ ] **E016** - The number of sequences and waveplate positions should be validated as a function of the provided `WAVEPLATE_POS` parameter.
- [ ] **E017** - The image readout times should be validated as a function of the operation mode
- [ ] **E018** - When verifying the waveplate condition, S4ACS should log a warning if, after 1.5 s of the exposure request, the waveplate stills not ready.
- [ ] **E019** - S4ACS should freeze the information to be written in the image header immediately before the start of the exposure
- [ ] **E020** - If the exposure effective time was smaller than `0.95 x texp`, a warning message should be logged and an error flag should be written in the image header.
- [ ] **E021** - A message should be logged after then end of the acquisition of an image series and after the end of writting the images to disk.
- [ ] **E022** - If S4ACS is run in the video mode, no image should be created after an exposure.

----------------------------------------------------------------------------------------

# When S4ACS is stopped
- [ ] **C001** - Before shutting the camera down, its shutter should be closed and the cooler should be turned off.
- [ ] **C002** - The log files should be closed


