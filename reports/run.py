from pathlib import Path

import pandas as pd
from astropy.io import fits
from data_types import Header_Content, S4ACS_Config
from utils import create_csv

night = "20260729"
csv_file = Path.cwd() / "_csv" / f"{night}.csv"
create_csv(csv_file)
config = S4ACS_Config.from_config_file("acs_config.cfg")
images_folder = config.image_path / ".." / night
images_list = [
    name.resolve() for name in images_folder.iterdir() if name.suffix == ".fits"
]


for image_file in images_list:
    hdr = fits.getheader(image_file)
    hdr_cnt = Header_Content.from_header(hdr)
    df = pd.DataFrame([hdr_cnt.to_csv_format])
    df.to_csv(csv_file, mode="a", index=False, header=False)
