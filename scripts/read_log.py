#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script Name: read_log.py
Description: Read the S4ACS events log file of the last night, looking for error logs.

Author: Denis Bernardes
Date: 2025-06-02
Version: 1.0

Usage:
    python read_log.py

"""

import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import dotenv
import utils

logger = logging.getLogger(__name__)

logger.info(f"The python interpreter used in this run is {sys.executable}")


# --------- Read CFG file ---------------
acs_config = utils.S4ACS_Config.from_config_file("acs_config.cfg")
logger.info("The config file has been read")
logger.info(f"This machine correspons to ACS{acs_config.channel}.")
logger.info(f"The path in which the log files are saved is {acs_config.image_path}.")

# ------------------- Get yesterday str -------------------
yesterday = datetime.now() - timedelta(days=1)
yesterday = yesterday.strftime("%Y%m%d")
logger.info(f"The observation date was {yesterday}.")

# --------- Read log file ---------------
log_file = acs_config.log_file_path / f"{yesterday}_events.log"
if not log_file.exists():
    logger.error(f"File {log_file} not found")
    sys.exit()
with open(log_file) as file:
    lines = [line for line in file.read().splitlines() if "ERROR" in line]
logger.info("The log file has been read")
len_lines = len(lines)
if len_lines == 0:
    logger.info("There is no error to report")
    logger.info("Exiting the script")
    sys.exit()
logger.info(f"There is (are) {len_lines} line(s) to log")


BASE_STRING = f"Hello,\
\nYou are receiving the ACS errors found for the SPARC4 channel {acs_config.channel}, \
for the observation night {yesterday}.\n\n"
for line in lines:
    BASE_STRING += line + "\n"


# ------------ Send email --------------------
dotenv.load_dotenv()
USER = os.getenv("GMAIL_USER")
PASSWORD = os.getenv("GMAIL_KEY")
RECEIVERS = [USER]
msg = MIMEMultipart()
msg["From"] = USER  # type: ignore
msg["Subject"] = (
    f"{yesterday}: errors found for the SPARC4 channel {acs_config.channel}."
)
msg.attach(MIMEText(BASE_STRING, "plain"))
try:
    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    server.login(USER, PASSWORD)  # type: ignore
    texto = msg.as_string()
    for receiver in RECEIVERS:
        msg["To"] = receiver  # type: ignore
        server.sendmail(USER, receiver, texto)  # type: ignore
        logger.info(f"The email has been sent to {receiver} succesfully.")
    server.quit()
except Exception as e:
    logger.info(f"Error when sending the email: {repr(e)}")
