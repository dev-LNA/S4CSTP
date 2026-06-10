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
from os.path import join
from pathlib import Path

import dotenv

cwd = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    filename=join(cwd, "log.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info(f"The python interpreter used in this run is {sys.executable}")


# --------- Read CFG file ---------------
username = cwd.split("\\")[2]
config_file = Path(f"C:/Users/{username}/SPARC4/ACS") / "acs_config.cfg"
if not config_file.exists():
    logging.error(f"file {config_file} not found")
    sys.exit()
config = configparser.ConfigParser()
config.read(config_file)
logging.info(f"The file {config_file} has been read.")
channel = config.get("channel configuration", "channel")
logging.info(f"This machine correspons to ACS{channel}.")
log_folder = config.get("channel configuration", "log file path").replace('"', "")
log_folder = Path(log_folder)
logging.info(f"The path in which the log files are saved is {log_folder}.")

# --------- Read the log file ---------------
yesterday = datetime.now() - timedelta(days=1)
yesterday = yesterday.strftime("%Y%m%d")
logging.info(f"The observation date was {yesterday}.")
log_file = log_folder / f"{yesterday}_events.log"
if not log_file.exists():
    logging.error(f"file {log_file} not found")
    sys.exit()
with open(log_file) as file:
    lines = file.read().splitlines()
logging.info("The log file has been read.")

BASE_STRING = f"""
Hello,

You are receiving the errors occurred in {yesterday}, found for the SPARC4 channel {channel}.

"""
EMAIL_STRING = BASE_STRING
i = 0
for line in lines:
    if "ERROR" in line:
        EMAIL_STRING += line + "\n"
        i += 1
logging.info(f"There is (are) {i} line(s) to log.")
if i == 0:
    logging.info("Exiting the script.")
    sys.exit()

# ------------ Send email --------------------
dotenv.load_dotenv()
USER = os.getenv("GMAIL_USER")
PASSWORD = os.getenv("GMAIL_KEY")
RECEIVERS = [USER]
msg = MIMEMultipart()
msg["From"] = USER  # type: ignore
msg["Subject"] = f"{yesterday}: errors found for the SPARC4 channel {channel}."
msg.attach(MIMEText(EMAIL_STRING, "plain"))
try:
    server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    server.login(USER, PASSWORD)  # type: ignore
    texto = msg.as_string()
    for receiver in RECEIVERS:
        msg["To"] = receiver  # type: ignore
        server.sendmail(USER, receiver, texto)  # type: ignore
        logging.info(f"The email has been sent to {receiver} succesfully.")
    server.quit()
except Exception as e:
    logging.info(f"Error when sending the email: {repr(e)}")
