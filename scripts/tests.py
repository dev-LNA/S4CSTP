import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

file = "SDK 2.104.30000.0"
# file = "SDK 2.104.30167.0"
image = Image.open(
    rf"C:\Users\Denis\Downloads\20260701_tests\20260701_tests\{file}.tiff"
)
image = np.array(image)
plt.plot(image[500, :] / 2, "r-", label="TIFF - I16")


image = fits.getdata(
    rf"C:\Users\Denis\Downloads\20260701_tests\20260701_tests\{file}.fits"
)
image = np.array(image)
plt.plot(image[500, :], "b-", label="FITS - U16")

image = fits.getdata(
    r"C:\Users\Denis\Downloads\20260701_tests\20260701_tests\andor_solis.fits"
)
image = np.array(image)
plt.plot(image[0][500, :], "g-", label="Andor Solis")


plt.legend()
plt.xlabel("X axis (pix)")
plt.ylabel("Counts (ADU)")
plt.show()
