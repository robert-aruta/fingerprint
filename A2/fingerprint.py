
from os import path
import utils  # Run utils.py for helper functions
import subprocess
import math
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils import *
from ipywidgets import interact

if not path.exists('utils.py'): # If running on colab: the first time download and unzip additional files
    
    # Run the wget command to download the file
    subprocess.run(['wget', 'https://biolab.csr.unibo.it/samples/fr/files.zip'])

    # Run the unzip command to extract the contents
    subprocess.run(['unzip', 'files.zip'])

# Read Fingerprints in the fingerprint subdirectory
def read_fingerprints():

    directoryPath = './A2/DB1_B/'
    fingerprints = []
    resizedImages = []

    for filename in os.listdir(directoryPath):
        
        filepath = os.path.join(directoryPath, filename)
        image = cv.imread(filepath)

        if fingerprints is not None:

            fingerprints.append(image)

        else:
            print("Failed to Read image of %s\n" % filename)

    return fingerprints

# Calculate the local gradient for all the fingerprins in the directory
def calc_sobel(fingerprints):
    
    GxList = []
    GyList = []

    # Calculate the local gradient for each fingerprins in the directory 
    for f in fingerprints:
        
        Gx = cv.Sobel(f, cv.CV_32F, 1, 0)
        Gx = Gx**2
        GxList.append(Gx)
        
        Gy = cv.Sobel(f, cv.CV_32F, 0, 1)
        Gy = Gy**2
        GyList.append(Gy)
    
    Gm = [np.sqrt(x + y) for x, y in zip(GxList, GyList)]
    
    return Gm

# # Get the current working directory
# current_directory = os.getcwd()

# # Print the current working directory
# print("Current Working Directory:", current_directory)
    
readFingerprints = read_fingerprints()
fingerprintSobel = calc_sobel(readFingerprints)
cv.imshow('Sobel', fingerprintSobel[0])
# for i, fpS in enumerate(fingerprintSobel):
#     cv.imshow('Sobel %d' % i, fpS)