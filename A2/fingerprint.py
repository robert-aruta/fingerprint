from os import path
import subprocess

if not path.exists('utils.py'): # If running on colab: the first time download and unzip additional files
    
    # Run the wget command to download the file
    subprocess.run(['wget', 'https://biolab.csr.unibo.it/samples/fr/files.zip'])

    # Run the unzip command to extract the contents
    subprocess.run(['unzip', 'files.zip'])

import utils  # Run utils.py for helper functions
import math
import os
import subprocess
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from utils import *
from ipywidgets import interact
from PyQt5.QtWidgets import *



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

# Calculate the local gradient for all the fingerprints in the directory
def calc_sobel(fingerprints):
    
    GxList = []
    GyList = []
    Gx2List = []
    Gy2List = []

    # Calculate the local gradient for each fingerprins in the directory 
    for image, filename in fingerprints:
        
        # Applies the Sobel Filter to the image and append it to the list
        # for the x-direction
        Gx = cv.Sobel(image, cv.CV_32F, 1, 0)
        GxList.append(Gx)
         
        # Squares the output filtered image and appends it to the list for the
        # x-direction
        Gx2 = Gx**2
        Gx2List.append(Gx2)
        
        # Applies the Sobel Filter to the image and append it to the list
        # for the y-direction
        Gy = cv.Sobel(image, cv.CV_32F, 0, 1)
        GyList.append(Gy)

        # Squares the output filtered image and appends it to the list for the
        # y-direction
        Gy2 = Gy**2
        Gy2List.append(Gy2)
    
    
    GmList = [np.sqrt(x + y) for x, y in zip(Gx2List, Gy2List)]
    
    return GxList, GyList, Gx2List, Gy2List, GmList


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

# Calculates the estimation of the local ridge orientation
def ridge_orientation(GxList, GyList, Gx2List, Gy2List):
    
    W = (23, 23)
    GxxList = []
    GyyList = []
    GxyList = []
    orientationsList = []
    strengthsList = []
    
    # Applies boxfilter for all fingerprints in the X Gradient List
    for gx2 in Gx2List:
        
        gxx = cv.boxFilter(gx2, -1, W, normalize = False)
        GxxList.append(gxx)
    
    # Applies boxfilter for all fingerprints in the Y Gradient List
    for gy2 in Gy2List:
        
        gyy = cv.boxFilter(gy2, -1, W, normalize = False)
        GyyList.append(gyy)
    
    # Applies boxfilter after multiplying X and Y gradients
    for gx, gy in zip(GxList, GyList):
        
        gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
        GxyList.append(gxy)
    
    # Substracts each X and Y gradients after being box filtered    
    diffGxxGyyList = [gxx - gyy for gxx, gyy in zip(GxxList, GyyList)]
    
    # Multiplies each element in the GxyList by 2
    G2xyList = [2 * gxy for gxy in GxyList]
    
    # Sums each X and Y gradients after being box filtered
    sumGxxGyyList = [gxx + gyy for gxx, gyy in zip(GxxList, GyyList)]
    
    
    for diffGxxGyy, g2xy in zip(diffGxxGyyList, G2xyList):
        
        orientations = (cv.phase(diffGxxGyy, -g2xy) + np.pi) / 2 # '-' to adjust for y axis direction
        orientationsList.append(orientations)
        
    
    
    for diffGxxGyy, g2xy, sumGxxGyy, gxx in zip(diffGxxGyyList, G2xyList, sumGxxGyyList, GxxList):
        
        strengths = np.divide(cv.sqrt(((diffGxxGyy ** 2) + (g2xy ** 2))), sumGxxGyy, out=np.zeros_like(gxx), where=sumGxxGyy !=0)
        strengthsList.append(strengths)
    
    return orientationsList, strengthsList

def ridge_frequency(fingerprints):
    
    regionList = []
    blurRegionList = []
    xSigList = []
    locMaxList = []
    
    for fingerprint, filename in fingerprints:
        
        #region = fingerprint[100:20, 50:150]
        # w, h = fingerprint.shape

        # startX = (w - cropWidth) // 2
        # startY = (h - cropHeight) // 2
        # endX = startX - cropWidth
        # endY = startY - cropHeight
        
        # region = fingerprint[startX:endX, 50:endY]
        # regionList.append(region)
        w, h = fingerprint.shape
        cropWidth = int(0.3 * w)
        cropHeight = int(0.3 * h)
        endRow = (h - cropHeight) // 2 
        startRow = endRow - 80
        cropWidth = int(0.3 * w)
        cropHeight = int(0.3 * h)
        middleColumn = (w - cropWidth) // 2
        startColumn = middleColumn - 25
        endColumn = middleColumn + 25
        
        region = fingerprint[startRow:endRow, startColumn:endColumn]
        regionList.append(region)
        # endX = startX + 80
        # endY = (h - cropHeight) // 2
        # endY = startY - cropHeight
        
        blurRegion = cv.blur(region, (5,5), -1)
        blurRegionList.append(blurRegion)
        
        xSignature = np.sum(blurRegion, 1)
        xSigList.append(xSignature)
    
    for xs in xSigList:
        
        localMax = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
        locMaxList.append(localMax)
        
    return regionList, locMaxList, xSigList

def ridge_period(locMaxList):
    
    distList = []
    ridgePeriodList = []
    
    for locMax in locMaxList:
        
        distance = locMax[1:] - locMax[:-1]
        distList.append(distance)
        
    for distance in distList:
        
        ridgePeriod = np.average(distance)
        ridgePeriodList.append(ridgePeriod)
        
    return distList, ridgePeriodList

def gabor_bank(ridgePeriodList):
    
    orCount = 8
    gaborBankList = []
    
    for ridgePeriod in ridgePeriodList:
        
        gaborBank = [gabor_kernel(ridgePeriod, o) for o in np.arange(0, np.pi, np.pi/orCount)]
        gaborBankList.append(gaborBank)
    
    return gaborBankList
def print_wd():
    # Get the current working directory
    currentDirectory = os.getcwd()

    # Print the current working directory
    print(currentDirectory)

def gui_init():
    
    #print("inside GUI INIT function")
    app = QApplication([])
    window = QWidget()
    
    label = QLabel(window)
    label.setText("Hello World")
    
    window.show()
    app.exec_()
    
def main():
    pass
    # print_wd()
    # readFingerPrints = read_fingerprints()
    # GxList, GyList, Gx2List, Gy2List, GmList = calc_sobel(readFingerPrints)
    # Gx = GxList[0]
    # _, filename = readFingerPrints[0]
    # sumGm = sum_Gm(GmList)
    # thres = threshold_mask(sumGm)
    # plt.imshow(thres[0], cmap='gray')  # You can specify the colormap to 'gray' for grayscale images
    # plt.title(f'Integral Gradient Magnitude of {filename}')
    # plt.show()
    # orientationsList, strengthsList = ridge_orientation(Gx2List, Gy2List)
    
    # cv.imshow('', orientationsList[0])
    
    # while True:

    #     if cv.waitKey(1) & 0xFF == ord('q'):
            
    #         cv.destroyAllWindows()
    #         break
        
if __name__ == "__main__":
    main()


