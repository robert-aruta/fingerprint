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
            
            matchList.append(score)
            
        else:
            
            notMatchList.append(score)
            
    if TPList and FPList is not None:
        
        TPR = len(matchList) / (len(matchList) + len(notMatchListt)) 
        
    return TPR

    
    
# Compare the uploaded fingerprint against the database
def compare_fingerprints(whatevertheoutputofaanalyse_target_fingerprintis):
    pass

def main():
    
    st.title('Fingerprint Analysis Tool')
    # File path for target fingerprints to be analysed by the user
    directoryPath = './target_fingerprint'
    
    # Initialise fingerprint database
    fingerprintsList, validMinutiaeList, localStructsList = \
        fp.analyse_fingerprints(None)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        
        st.image(fingerprintsList[0][0], caption = 'Fingerprint', use_column_width = True)
    
    with col2:
        
        drawMinutiae = draw_minutiae(fingerprintsList[0][0], validMinutiaeList[0])
        st.image(drawMinutiae, caption = 'Fingerprint', use_column_width = True)
        
    with col3:
        pass
        #st.image(localStructsList[0], caption = 'Fingerprint', use_column_width = True)
   


    uploadedFile = st.file_uploader("Upload fingerprint image...", \
        type = ["png", "jpg", "jpeg", "tif"])

    if uploadedFile is not None:
        
        filePath = os.path.join(directoryPath, uploadedFile.name)
        
        with open(filePath, "wb") as f:
            
            f.write(uploadedFile.getbuffer())
        
        st.success('File Successfuly Uploaded')
        f1, m1, ls1 = fp.analyse_fingerprints(filePath)
        #st.image(f)
        # move the uploaded file to the fingerprint database directory
        newFilePath = os.path.join('./DB1_B', uploadedFile.name)
        os.rename(filePath, newFilePath)
        #st.success('File Moved to DB1_B directory')
        encodeFile = np.asarray(bytearray(uploadedFile.read()), dtype=np.uint8)
        decodeFile = cv.imdecode(encodeFile, cv.IMREAD_GRAYSCALE)
        #st.image(decodeFile, channels="BGR", caption='Uploaded Fingerprint', use_column_width=True)
        #print(decodeFile.min(). decodeFile.max())
        #analyse_target_fingerprint(decodeFile)
        
        
        
if __name__ == "__main__":
    main()
    