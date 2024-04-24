from os import path
import subprocess
import threading
import streamlit as st
import cv2 as cv

if not path.exists('utils.py'): # If running on colab: the first time download and unzip additional files
    
    # Run the wget command to download the file
    subprocess.run(['wget', 'https://biolab.csr.unibo.it/samples/fr/files.zip'])

    # Run the unzip command to extract the contents
    subprocess.run(['unzip', 'files.zip'])

import utils  # Run utils.py for helper functions
import fingerprint as fp
import math
import os
import subprocess
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import streamlit as st
from utils import *
from ipywidgets import interact


###################
# Run the fingerprint analysis methodology on the fingerprint database
def fingerprint_db_init():
    
    fingerprintsList = fp.read_fingerprints()
    GxList, GyList, Gx2List, Gy2List, GmList = fp.calc_sobel(fingerprintsList)
    sumGm = fp.sum_Gm(GmList)
    maskList = fp.threshold_mask(sumGm)
    orientationsList, strengthsList = fp.ridge_orientation(GxList, GyList,\
        Gx2List, Gy2List)
    
    regionList, locMaxList, xSigList = fp.ridge_frequency(fingerprintsList)
    distanceList, ridgePeriodList = fp.ridge_period(locMaxList)
    gaborBankList = fp.gabor_bank(ridgePeriodList)
    fFPList, nfFBList = fp.filter_fingerprint(fingerprintsList, gaborBankList)
    enhancedFPList = fp.enhance_fingerprint(fingerprintsList, fFPList, \
        orientationsList, maskList)
    
    ridgeLinesList = fp.ridge_lines(enhancedFPList)
    skeletonList = fp.get_skeleton(ridgeLinesList)
    neighbourValsList, allEightNeighbors, cnList = fp.get_cn(skeletonList)
    minutiaeList = fp.get_minutiae(cnList)
    maskDistList = fp.get_mask_distance(maskList)
    filtMinutiaeList = fp.get_filt_minutiae(maskDistList, minutiaeList)
    xySteps, ndLUT = fp.create_nd_LUT(allEightNeighbors)
    classMinutiaeDir = fp.MinutiaeDirections(neighbourValsList, ndLUT, \
        cnList, xySteps)
    
    validMinutiaeList = classMinutiaeDir.valid_minutiae(filtMinutiaeList)
    
    print("FINGERPRINT DATABASE INITIALIZED")
    
    return 0

# Label the fingerprint based on the score
def label_fingerprint(scoresList):
    
    matchList = [] 
    notMatchList = []
    
    for score in scoresList:
        
        if score >= 0.5:
            
            TPList.append(score)
            
        else:
            
            FNList.append(score)
            
    if TPList and FPList is not None:
        
        TPR = len(TPList) / (len(TPList) + len(FPList)) 
        
    return TPR

# Normalize image values to be displayed in streamlit
def normalize_values(values):
    
    normValues = (values - values.min()) / (values.max() - values.min())
    
    return normValues

def follow_ridge_and_compute_angle(neighborhoodVaues, x, y, d = 8):
    
    px, py = x, y
    length = 0.0
    
    while length < 20: # max length followed
        nextDirections = ndLUT[neighborhoodVaues[py,px]][d]
        #print("line 6")
        if len(nextDirections) == 0:
            print("line 7")
            break
        # Need to check ALL possible next directions
        if (any(cn[py + xy_steps[nd][1], px + xy_steps[nd][0]] != 2 for nd in next_directions)):
            print("line 11")
            break # another minutia found: we stop here
        # Only the first direction has to be followed
        d = nextDirections[0]
        ox, oy, l = xy_steps[d]
        px += ox ; py += oy ; length += l
        print("length = %d" % length)
    # check if the minimum length for a valid direction has been reached
    return math.atan2(-py+y, px-x) if length >= 10 else None


# # Run the fingerprint methodology for the uploaded fingerprint
# def analyse_target_fingerprint(uploadedFile):
    
#     fingerprint = uploadedFile
#     gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1)
#     gx2, gy2 = gx**2, gy**2
#     gm = np.sqrt(gx2 + gy2)
#     sumGm = cv.boxFilter(gm, -1, (25, 25), normalize = False)
#     thr = sumGm.max() * 0.2
#     mask = cv.threshold(sumGm, thr, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
#     normMask = normalize_values(mask)
#     merge = cv.merge((mask, fingerprint, fingerprint))

#     # Estimation of local ridge orientation
#     W = (23, 23)
#     gxx = cv.boxFilter(gx2, -1, W, normalize = False)
#     gyy = cv.boxFilter(gy2, -1, W, normalize = False)
#     gxy = cv.boxFilter(gx * gy, -1, W, normalize = False)
#     GxxGyy = gxx - gyy
#     gxy2 = 2 * gxy
#     orientations = (cv.phase(GxxGyy, -gxy2) + np.pi) / 2 # '-' to adjust for y axis direction
#     sumGxxGyy = gxx + gyy
#     strengths = np.divide(cv.sqrt((GxxGyy**2 + gxy2**2)), sumGxxGyy, \
#         out=np.zeros_like(gxx), where=sumGxxGyy!=0)
    
#     drawOrientations = draw_orientations(fingerprint, orientations, strengths, mask, 1, 16)
#     region = fingerprint[10:90,80:130]
    
#     # before computing the x-signature, the region is smoothed to reduce noise
#     smoothed = cv.blur(region, (5,5), -1)
#     xs = np.sum(smoothed, 1) # the x-signature of the region
#     localMaxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
#     distances = localMaxima[1:] - localMaxima[:-1]
#     ridgePeriod = np.average(distances)
    
#     orCount = 8
#     gaborBank = [gabor_kernel(ridgePeriod, o) for o in np.arange(0, np.pi, np.pi/orCount)]
#     nf = 255 - fingerprint
#     allFiltered = np.array([cv.filter2D(nf, cv.CV_32F, f) for f in gaborBank])
#     yCoords, xCoords = np.indices(fingerprint.shape)
#     orientationIdx = np.round(((orientations % np.pi) / np.pi) * orCount).astype(np.int32) % orCount
#     filtered = allFiltered[orientationIdx, yCoords, xCoords]
#     enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
    
#     _, ridgeLines = cv.threshold(enhanced, 32, 255, cv.THRESH_BINARY)
#     skeleton = cv.ximgproc.thinning(ridgeLines, thinningType = cv.ximgproc.THINNING_GUOHALL)
#     cnFilter = fp.create_cn_filter()
#     allEightNeighbors = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
#     cnLUT = np.array([fp.compute_crossing_number(x) for x in allEightNeighbors]).astype(np.uint8)
    
#     # Skeleton: from 0/255 to 0/1 values
#     skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
    
#     # Apply the filter to encode the 8-neighborhood of each pixel into a byte [0,255] 
#     neighborhoodValues = cv.filter2D(skeleton01, -1, cnFilter, borderType = cv.BORDER_CONSTANT)
    
#     # Apply the lookup table to obtain the crossing number of each pixel from the byte value of its neighborhood
#     cn = cv.LUT(neighborhoodValues, cnLUT)
    
#     # Keep only crossing numbers on the skeleton
#     cn[skeleton == 0] = 0
    
#     # crossing number == 1 --> Termination, crossing number == 3 --> Bifurcation
#     minutiae = [(x1,y1,cn[y1,x1]==1) for y1, x1 in zip(*np.where(np.isin(cn, [1,3])))]
    
#     drawMinutiaeFP = draw_minutiae(fingerprint, minutiae)
#     drawMinutiaeSkel = draw_minutiae(skeleton, minutiae)
#     r2 = 2**0.5 # sqrt(2)
    
#     # The eight possible (x, y) offsets with each corresponding Euclidean distance
#     xySteps = [(-1,-1,r2),( 0,-1,1),( 1,-1,r2),( 1, 0,1),( 1, 1,r2),( 0, 1,1),(-1, 1,r2),(-1, 0,1)]

#     # LUT: for each 8-neighborhood and each previous direction [0,8], 
#     #      where 8 means "none", provides the list of possible directions
#     ndLUT = [[compute_next_ridge_following_directions(pd, x) for pd in \
#         range(9)] for x in all_8_neighborhoods]
    
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
        
#         st.image(fingerprint, caption='Gx**2', use_column_width=True)
    
#     with col2:
        
#         st.image(drawMinutiaeFP, caption='Gy**2', use_column_width=True)
    
#     with col3:
        
#         st.image(drawMinutiaeSkel, caption='Gradient Magnitude', use_column_width=True)
        
# Compare the uploaded fingerprint against the database
def compare_fingerprints(whatevertheoutputofaanalyse_target_fingerprintis):
    pass

def main():
    st.title('Fingerprint Analysis Tool')
    
    fingerprint_db_init()
    #fp.print_wd()
    uploadedFile = st.file_uploader("Upload fingerprint image...", type = ["png", "jpg", "jpeg", "tif"])

    if uploadedFile is not None:
        
        encodeFile = np.asarray(bytearray(uploadedFile.read()), dtype=np.uint8)
        decodeFile = cv.imdecode(encodeFile, cv.IMREAD_GRAYSCALE)
        #st.image(decodeFile, channels="BGR", caption='Uploaded Fingerprint', use_column_width=True)
        #print(decodeFile.min(). decodeFile.max())
        analyse_target_fingerprint(decodeFile)
        
        
        
if __name__ == "__main__":
    main()
    