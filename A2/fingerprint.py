from os import path
import subprocess
import threading

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
import csv
import matplotlib.pyplot as plt
import streamlit as st
from utils import *
from ipywidgets import interact
#from PyQt5.QtWidgets import *



# Read Fingerprints in the fingerprint subdirectory
def read_fingerprints():

    directoryPath = './DB1_B/'
    fingerprintList = []
    resizedImages = []

    for filename in os.listdir(directoryPath):
        
        filepath = os.path.join(directoryPath, filename)
        image = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

        if fingerprintList is not None:

            fingerprintList.append((image, filename))

        else:
            print("Failed to Read image of %s\n" % filename)

    # with open('output.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     # Write all rows at once
    #     writer.writerows(fingerprintList)
    
    return fingerprintList

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
        
    
    
    for diffGxxGyy, g2xy, sumGxxGyy, gxx in zip(diffGxxGyyList, G2xyList, \
        sumGxxGyyList, GxxList):
        
        strengths = np.divide(cv.sqrt(((diffGxxGyy ** 2) + (g2xy ** 2))), \
                              sumGxxGyy, out=np.zeros_like(gxx), \
                                  where=sumGxxGyy !=0)
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
        
        localMax = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & \
            np.r_[xs[:-1] >= xs[1:], False])[0]
        
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

def filter_fingerprint(fingerprints, gaborBankList):
    
    filteredFPList = []
    nonfilteredFPList = []
    
    for (fp, filename), gaborBank in zip(fingerprints, gaborBankList):
        
        nfFP = 255 - fp
        nonfilteredFPList.append(nfFP)
        
        fFP = np.array([cv.filter2D(nfFP, cv.CV_32F, gb) for gb in gaborBank])
        filteredFPList.append(fFP)
        
    return filteredFPList, nonfilteredFPList 

def enhance_fingerprint(fingerprintsList, filteredFPList, orientationsList, maskList):
    
    enhancedFPList = []
    xList = []
    yList = []
    orientationIdxList = []
    filteredList = []
    orCount = 8
    
    for fingerprint, filename in fingerprintsList:
        
        y, x = np.indices(fingerprint.shape)
        xList.append(x)
        yList.append(y)
    
    for orientations in orientationsList:
        
        orientationIdx = np.round(((orientations % np.pi) / np.pi) * orCount).astype(np.int32) % orCount
        orientationIdxList.append(orientationIdx)
    
    for x, y, fFp, orientationIdx in zip(xList, yList, filteredFPList, orientationIdxList):
        
        filteredFP = fFp[orientationIdx, y, x]
        filteredList.append(filteredFP)
    
    for mask, filteredFP in zip(maskList, filteredList):
        
        enhancedFP = mask & np.clip(filteredFP, 0, 255).astype(np.uint8)
        enhancedFPList.append(enhancedFP)
    # for fFp, orientations, mask in zip(filteredFPList, orientationsList, maskList):
        
    #     orientationIdx = np.round(((orientations % np.pi) / np.pi) * orCount).astype(np.int32) % orCount
    #     filterIdx = fFp[orientationIdx, y, x]
    #     enhancedFP = mask & np.clip(filterIdx, 0, 255).astype(np.unit8)
    #     enhancedFPList.append(enhancedFP)
        
    return enhancedFPList

def ridge_lines(enhancedFPList):
    
    ridgeLinesList = []
    
    for enhancedFP in enhancedFPList:
    
        _, ridgeLines = cv.threshold(enhancedFP, 32, 255, cv.THRESH_BINARY)
        ridgeLinesList.append(ridgeLines)
    
    return ridgeLinesList

def get_skeleton(ridgeLinesList):
    
    skeletonList = []
    
    for ridgeLines in ridgeLinesList:
        
        skeleton = cv.ximgproc.thinning(ridgeLines, thinningType = cv.ximgproc.THINNING_GUOHALL)
        skeletonList.append(skeleton)
    
    return skeletonList

def compute_crossing_number(values):
    return np.count_nonzero(values < np.roll(values, -1))

def get_cn(skeletonList):
    
    cnFilter = np.array([[  1,  2,  4],
                      [128,  0,  8],
                      [ 64, 32, 16]
                     ])
    
    allEightNeighbors = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
    cnLUT = np.array([compute_crossing_number(x) for x in allEightNeighbors]).astype(np.uint8)
    
    skeleton01List = []
    
    for skeleton in skeletonList:
        
        skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
        skeleton01List.append(skeleton01)
    
    neighbourValsList = []
    
    for skeleton01 in skeleton01List:
        
        neighbourVals = cv.filter2D(skeleton01, -1, cnFilter, borderType = cv.BORDER_CONSTANT)
        neighbourValsList.append(neighbourVals)
    
    cnList = []
    
    for skeleton, neighbourVals in zip(skeletonList, neighbourValsList):
        
        cn = cv.LUT(neighbourVals, cnLUT)
        cn[skeleton == 0] = 0
        cnList.append(cn)
        
    return neighbourValsList, allEightNeighbors, cnList

def get_minutiae(cnList):
    
    minutiaeList = []
    
    for cn in cnList:
        
        minutiae = [(x, y, cn[y, x]==1) for y, x in zip(*np.where(np.isin(cn, [1,3])))]
        minutiaeList.append(minutiae)
    
    return minutiaeList

def get_mask_distance(maskList):
    
    maskDistList = []
    
    for mask in maskList:
        
        maskDist = cv.distanceTransform(
            cv.copyMakeBorder(mask, 1, 1, 1, 1, 
                              cv.BORDER_CONSTANT), cv.DIST_C, 3)[1:-1,1:-1]
        
        maskDistList.append(maskDist)

    return maskDistList

def get_filt_minutiae(maskDistList, minutiaeList):
    
    filtMinutiaeList = []
    
    for maskDist, minutiae in zip(maskDistList, minutiaeList):
        
        filteredMinutiae = list(filter(lambda m: maskDist[m[1], m[0]]>10,
                                       minutiae))
        
        filtMinutiaeList.append(filteredMinutiae)
        
    return filtMinutiaeList

def compute_next_ridge_following_directions(prevDirection, values):
        
    nextPos = np.argwhere(values!=0).ravel().tolist()
    
    if len(nextPos) > 0 and prevDirection != 8:
        # There is a previous direction: return all the next directions, sorted
        # according to the distance from it, except the direction, if any, 
        # that corresponds to the previous position
                       
        nextPos.sort(key = lambda d: 4 - abs(abs(d - prevDirection) - 4))
        
        if nextPos[-1] == (prevDirection + 4) % 8: # the direction of the previous position is the opposite one
            nextPos = nextPos[:-1] # removes it
            
    return nextPos

def create_nd_LUT(allEightNeighbors):
    
    r2 = 2**0.5 # sqrt(2)
    
    # The eight possible (x, y) offsets with each corresponding Euclidean distance
    xySteps = [(-1, -1, r2),(0, -1, 1),(1, -1, r2), (1, 0, 1), (1, 1, r2), 
               ( 0, 1, 1), (-1, 1, r2),(-1, 0, 1)]

    ndLUT = [[compute_next_ridge_following_directions(pd, x) for pd in 
              range(9)] for x in allEightNeighbors]
    
    return xySteps, ndLUT



class MinutiaeDirections:
    
    def __init__(self, neighbourValsList, ndLUT, cnList, xySteps):
        
        self.neighbourValsList = neighbourValsList
        self.ndLUT = ndLUT
        self.cnList = cnList
        self.xySteps = xySteps
        
    def follow_ridge_and_compute_angle(self, valTuple, x, y, d = 8):
        
        ndLUT = self.ndLUT
        xySteps = self.xySteps
        neighbourVals, cn = valTuple
        px, py = x, y
        length = 0.0

        # ndLUT = ridgeCtx.ndLUT
        # neighbourValsList = ridgeCtx.neighbourValsList
        # cnList = ridgeCtx.cnList
        # xySteps = ridgeCtx.xySteps
            
                
        while length < 20: # max length followed
            # print("length = %d" % length)
            # print("ndLUT: %s" % ndLUT)
            # print("py = %d" % py)
            # print("px = %d" % px)
            # print("d = %d" % d)
            #print("At start of loop - length: {}, px: {}, py: {}, d: {}"\
            #     .format(length, px, py, d))
            nextDirections = ndLUT[neighbourVals[py,px]][d]
            #print("nextDirections: {}".format(nextDirections))
            
            if len(nextDirections) == 0:
                
                #print("line 460")
                break
            
            # Need to check ALL possible next directions
            if (any(cn[py + xySteps[nd][1], px + xySteps[nd][0]] != 2 for nd 
                    in nextDirections)):
                #print("line 466")
                break # another minutia found: we stop here
            
            # Only the first direction has to be followed

            d = nextDirections[0]
            ox, oy, l = xySteps[d]
            #print("Before update - ox: {}, oy: {}, l: {}, px: {}, py: {}"\
            #     .format(ox, oy, l, px, py))
            px += ox ; py += oy ; length += l

            #print("After update - px: {}, py: {}, length: {}".\
                # format(px, py, length))

            # check if the minimum length for a valid direction has been reached

        return math.atan2(-py+y, px-x) if length >= 10 else None

    def valid_minutiae(self, filtMinutiaeList):
        
        #print("line 486")
        ndLUT = self.ndLUT
        neighbourValsList = self.neighbourValsList
        cnList = self.cnList
        xySteps = self.xySteps

        
        validMinutiaeList = []
        
        for filteredMinutiae, neighbourVals, cn \
            in zip(filtMinutiaeList, neighbourValsList, cnList):
                
            validMinutiae = []
            
            for x, y, term in filteredMinutiae:
                

                d = None
        
                if term: # termination: simply follow and compute the direction
                    
                    #print("line 503")
                    d = self.follow_ridge_and_compute_angle((neighbourVals, cn)
                                                       , x, y)

                    
                else: # bifurcation: follow each of the three branches
                    
                    # 8 means: no previous direction
                    
                    dirs = ndLUT[neighbourVals[y,x]][8]
                        
                    if len(dirs)==3: # only if there are exactly three branches
                        
                        #print("line 516")
                        angles = [self.follow_ridge_and_compute_angle \
                                ((neighbourVals, cn), x + xySteps[d][0], \
                                    y + xySteps[d][1], d) for d in dirs]

                        if all(a is not None for a in angles):
                            
                            a1, a2 = min(((angles[i], \
                                        angles[(i + 1) % 3]) \
                                        for i in range(3)), key=lambda \
                                        t: angle_abs_difference(t[0], t[1]))
                            
                            
                            d = angle_mean(a1, a2)  
                                        
                if d is not None:
                    validMinutiae.append( (x, y, term, d) )

                validMinutiaeList.append(validMinutiae)
                
        # with open('Valid Minutiae.csv', 'w', newline='') as file:
        #     writer = csv.writer(file)
        #     # Write all rows at once
        #     writer.writerows(validMinutiaeList)
            
        return validMinutiaeList
    
class LocalStructs:
    
    def __init__ (self, validMinutiaeList):
        
        self.mccRadius = 70
        self.mccSize = 16
        self.g = 2 * self.mccRadius / self.mccSize
        self.x = np.arange(self.mccSize) * self.g - (self.mccSize / 2) * \
            self.g + self.g / 2
        self.y = self.x[..., np.newaxis]
        self.iy, self.ix = np.nonzero(self.x**2 + self.y**2 <= \
            self.mccRadius**2)
        self.refCellCoords = np.column_stack((self.x[self.ix], self.x[self.iy]))
        self.mccSigmaS = 7.0
        self.mccTauPsi = 400.0
        self.mccMuPsi = 1e-2
        self.validMinutiaeList = validMinutiaeList
        
    def Gs(self, tSqr):
        # Gaussian function with zero mean and mcc_sigma_s standard deviation,
        # see eq. (7) in MCC paper
        return np.exp(-0.5 * tSqr / (self.mccSigmaS**2)) / (math.tau**0.5 * self.mccSigmaS)
    
    def Psi(self, v):
        # Sigmoid function that limits the contribution of dense minutiae 
        # clusters, see eq. (4)-(5) in MCC paper
        return 1. / (1. + np.exp(-self.mccTauPsi * (v - self.mccMuPsi)))
        
    def thread_calc_cell_coords(self, xy):
        
        cellCoords = np.transpose(rot@self.refCellCoords.T + \
                xy[:, :, np.newaxis], [0, 2, 1])
        
        return cellCords
    
    def create_local_structs(self):
        
        xydList = []
        
        for validMinutiae in self.validMinutiaeList:
            
            xyd = np.array([(x, y, d) for x, y, _, d in validMinutiae])
            xydList.append(xyd)
            
        dCosList = []
        dSinList = []
        
        for xyd in xydList:
            
            dCos, dSin = np.cos(xyd[:, 2]).reshape((-1, 1, 1)), \
                np.sin(xyd[:, 2]).reshape((-1, 1, 1))
                
            dCosList.append(dCos)
            dSinList.append(dSin)
            
        rotList = []
        
        for dCos, dSin in zip(dCosList, dSinList):
            
            rot = np.block([[dCos, dSin], [-dSin, dCos]])
            rotList.append(rot)
        
        xyList = []
        
        for xyd in xydList:
            
            xy = xyd[:, :2]
            xyList.append(xy)
            
        #xyArray = np.concatenate(xyList, axis = 0)
        
        localStructsList = []
        distsList = []
        
        for xy, rot in zip(xyList, rotList):
            
            cellCoords = np.transpose(rot@self.refCellCoords.T + \
                xy[:, :, np.newaxis], [0, 2, 1])

            dists = np.sum((cellCoords[:, :, np.newaxis, :] - xy) ** 2, -1)
            cs = self.Gs(dists)
            diagIndices = np.arange(cs.shape[0])
            cs[diagIndices, :, diagIndices] = 0
            localStructs = self.Psi(np.sum(cs, -1))
            localStructsList.append(localStructs)
            
        return localStructsList
    
class CompareFingerprint:
    
    def __init__ (self, fingerprintsList, validMinutiaeList, localStructsList):
        
        self.fingerprintsList = fingerprintsList
        self.validMinutiaeList = validMinutiaeList
        self.localStructsList = localStructsList
    
    def compare_fingerprints(self, target):
        pass
    
def print_wd():
    # Get the current working directory
    currentDirectory = os.getcwd()

    # Print the current working directory
    print(currentDirectory)

        
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


