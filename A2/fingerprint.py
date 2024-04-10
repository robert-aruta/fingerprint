from os import path

if not path.exists('utils.py'): # If running on colab: the first time download and unzip additional files
    
    # Run the wget command to download the file
    subprocess.run(['wget', 'https://biolab.csr.unibo.it/samples/fr/files.zip'])

    # Run the unzip command to extract the contents
    subprocess.run(['unzip', 'files.zip'])
    
import utils  # Run utils.py for helper functions
import math
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from utils import *
from ipywidgets import interact




# Read Fingerprints in the fingerprint subdirectory
def read_fingerprints():

    directoryPath = './A2/DB1_B/'
    fingerprints = []
    resizedImages = []

    for filename in os.listdir(directoryPath):
        
        filepath = os.path.join(directoryPath, filename)
        image = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

        if fingerprints is not None:

            fingerprints.append((image, filename))

        else:
            print("Failed to Read image of %s\n" % filename)

    return fingerprints

# Calculate the local gradient for all the fingerprins in the directory
def calc_sobel(fingerprints):
    
    GxList = []
    GyList = []

    # Calculate the local gradient for each fingerprins in the directory 
    for image, filename in fingerprints:
        
        Gx = cv.Sobel(image, cv.CV_32F, 1, 0)
        Gx = Gx**2
        GxList.append(Gx)
        
        Gy = cv.Sobel(image, cv.CV_32F, 0, 1)
        Gy = Gy**2
        GyList.append(Gy)
    
    GmList = [np.sqrt(x + y) for x, y in zip(GxList, GyList)]
    
    return GxList, GyList, GmList


def sum_Gm(GmList):
    
    sumList = []
    
    for g in GmList:
        
        sumGm = cv.boxFilter(g, -1, (25, 25), normalize = False)
        sumList.append(sumGm)
        
    return sumList

def threshold_mask(sumList):
    
    thresholdList = []
    maskList = []
    
    for i in sumList:
        
        threshold = i.max() * 0.2
        mask = cv.threshold(i, threshold, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
        maskList.append(mask)
        
    return maskList

#def orientation()
def print_wd():
    # Get the current working directory
    currentDirectory = os.getcwd()

    # Print the current working directory
    print(currentDirectory)

def main():
    
    print_wd()
    readFingerPrints = read_fingerprints()
    GxList, GyList, GmList = calc_sobel(readFingerPrints)
    Gx = GxList[0]
    _, filename = readFingerPrints[0]
    sumGm = sum_Gm(GmList)
    thres = threshold_mask(sumGm)
    plt.imshow(thres[0], cmap='gray')  # You can specify the colormap to 'gray' for grayscale images
    plt.title(f'Integral Gradient Magnitude of {filename}')
    plt.show()
    
if __name__ == "__main__":
    main()


