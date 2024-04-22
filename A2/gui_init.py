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
    
def normalize_values(values):
    
    normValues = (values - values.min()) / (values.max() - values.min())
    
    return normValues

# Run the fingerprint methodology for the uploaded fingerprint
def analyse_target_fingerprint(uploadedFile):
    
    fingerprint = uploadedFile
    gx, gy = cv.Sobel(fingerprint, cv.CV_32F, 1, 0), cv.Sobel(fingerprint, cv.CV_32F, 0, 1)
    normgx = normalize_values(gx)
    normgx = normalize_values(gy)
    st.image(normgx, channels="BGR", caption='Gx', use_column_width=True)
    
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
        decodeFile = cv.imdecode(encodeFile, cv.IMREAD_COLOR)
        #st.image(decodeFile, channels="BGR", caption='Uploaded Fingerprint', use_column_width=True)
        #print(decodeFile.min(). decodeFile.max())
        analyse_target_fingerprint(decodeFile)
        
        
        
if __name__ == "__main__":
    main()
    