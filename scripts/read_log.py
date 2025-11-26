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


import configparser
import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from getpass import getuser
from os.path import join
from pathlib import Path

from dotenv import load_dotenv

cwd = os.path.dirname(os.path.abspath(__file__))
base_folder = join(cwd, *3 * [".."], "SPARC4", "ACS")
logging.basicConfig(
    level=logging.INFO,
    filename=join(cwd, "log.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info(f"The python version used in this run is {sys.executable}")
# --------- Read CFG file ---------------
config = configparser.ConfigParser()
cfg_file = join(base_folder, "acs_config.cfg")
try:
    config.read(cfg_file)
    logging.info(f"The file {cfg_file} has been read.")
except FileNotFoundError as e:
    logging.info(f"The {cfg_file} file was not found." + repr(e))

channel = config.get("channel configuration", "channel")
logging.info(f"This machine correspons to ACS{channel}.")
log_folder = join(
    Path(config.get("channel configuration", "log file path").strip(r"\""))
)
logging.info(f"The path in which the log files are saved is {log_folder}.")

# --------- Read the log file ---------------
yesterday = datetime.now() - timedelta(days=1)
yesterday = yesterday.strftime("%Y%m%d")
logging.info(f"The observation date was {yesterday}.")
log_file = join(log_folder, f"{yesterday}_events.log")
try:
    with open(log_file) as file:
        lines = file.read().splitlines()
    logging.info(f"The log file has been read.")
except FileNotFoundError as e:
    logging.info(f"The file {yesterday}_events.log was not found." + repr(e))

BASE_STRING = f"""
Hello,

You are receiving the errors and warnings found for the SPARC4 channel {channel}, occurred in {yesterday}.

"""
EMAIL_STRING = BASE_STRING
i = 0
for line in lines:
    if "ERROR" in line:
        EMAIL_STRING += line + "\n"
        i += 1
logging.info(f"There is (are) {i} line(s) to log.")

# ------------ Send email --------------------
load_dotenv()
USER = os.getenv("GMAIL_USER")
PASSWORD = os.getenv("GMAIL_KEY")
RECEIVERS = [USER]


msg = MIMEMultipart()
msg["From"] = USER
msg["Subject"] = (
    f"{yesterday}: errors and warnings found for the SPARC4 channel {channel}."
)
msg.attach(MIMEText(EMAIL_STRING, "plain"))


if EMAIL_STRING != BASE_STRING:
    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(USER, PASSWORD)
        texto = msg.as_string()
        for receiver in RECEIVERS:
            msg["To"] = receiver
            server.sendmail(USER, receiver, texto)
            logging.info(f"The email has been sent to {receiver} succesfully.")
        server.quit()
    except Exception as e:
        logging.info(f"Error when sending the email: {repr(e)}")
else:
    logging.info(f"There are no events to report.")
    logging.info(f"There are no events to report.")
