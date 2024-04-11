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

# Calculate the local gradient for all the fingerprins in the directory
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

def ridge_orientation(GxList, GyList, Gx2List, Gy2List):
    
    W = (23, 23)
    GxxList = []
    GyyList = []
    GxyList = []
    orientationsList = []
    strengthsList = []
    diffGxxGyySqrlist = []
    G2xyListSqr = []
    
    for gx2 in Gx2List:
        
        gxx = cv.boxFilter(gx2, -1, W, normalize = False)
        GxxList.append(gxx)
    
    for gy2 in Gy2List:
        
        gyy = cv.boxFilter(gy2, -1, W, normalize = False)
        GyyList.append(gyy)
    
    
    for gx, gy in zip(GxList, GyList):
        
        gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
        GxyList.append(gxy)
          
    diffGxxGyyList = [gxx - gyy for gxx, gyy in zip(GxxList, GyyList)]
    G2xyList = [2 * gxy for gxy in GxyList]
    
    for diffgxxgyy, g2xy in zip(diffGxxGyyList, G2xyList):
        
        diffGxxGyySqr = diffgxxgyy ** 2
        diffGxxGyySqrlist.append(diffGxxGyySqr)
        
        G2xySqr = g2xy ** 2
        G2xyListSqr.append(G2xySqr)
        
    # for diffgxxgyy, g2xy in zip(diffGxxGyyList, G2xyList):
        
    #     orientations = (cv.phase(diffgxxgyy, -g2xy) + np.pi) / 2 # '-' to adjust for y axis direction
    #     orientationsList.append(orientations)
        
    # sumGxxGyyList = [gxx + gyy for gxx, gyy in zip(GxxList, GyyList)]
    
    # for diff_gxx_gyy, g2xy, sum_gxx_gyy, gxx in zip(diffGxxGyyList, G2xyList, sumGxxGyyList, GxxList):
        
    #     strengths = np.divide(cv.sqrt((diffgxxgyysqr + g2xysqr)), sum_gxx_gyy, out=np.zeros_like(gxx), where=sum_gxx_gyy !=0)
    #     strengthsList.append(strengths)
    
    # print("updated!")
    return GxyList         #orientationsList, strengthsList
    
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


