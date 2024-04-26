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
import pandas as pd
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

    
# Compare the uploaded fingerprint against the database
def compare_fingerprints(localStructsList, targetStructs): #databaseTuple, targetTuple
    
    ls1, targetName = targetStructs[0]
    scoresList = []
    
    for localStructs, filename in localStructsList:
        
        dists = np.linalg.norm(localStructs[:, np.newaxis, :] - ls1, axis = -1)
        
        # Normalize as in eq. (17) of MCC paper
        dists /= np.linalg.norm(localStructs, axis = 1)[:,np.newaxis] + \
            np.linalg.norm(ls1, axis = 1)
            
        numPairs = 5 # For simplicity: a fixed number of pairs
        pairs = np.unravel_index(np.argpartition(dists, numPairs, None)[:numPairs], \
            dists.shape)
        
        score = 1 - np.mean(dists[pairs[0], pairs[1]])
        scoresList.append((filename, score))
    
    # Convert scores to pandas Dataframe for streamlit
    scoreDataFrame = pd.DataFrame(scoresList, columns = ['Fingerprint', 'Score'])
    
    # Sort the table by Fingerprint name order
    scoreDataFrame.sort_values('Fingerprint', inplace = True)
    
    return scoreDataFrame

def create_ROC_curve(classifierDataFrame):
    
    filenameList = classifierDataFrame['Fingerprint']
    scoreList = classifierDataFrame['Score']
    classifierList = classifierDataFrame['Classifier']
    
    thresholdList = np.linspace(0, 1, 1001)
    TPList = []
    FPList = []
    TNList = []
    FNList = []
    TPRList = []
    FPRList = []
    
    for threshold in thresholdList:
        
        TP = FP = TN = FN = 0
        
        for file, classifier, score in zip(filenameList, classifierList, \
            scoreList):
        
            if score > threshold and classifier == 'True':
                
                TP += 1
            
            elif score > threshold and classifier == 'False':
                
                FP += 1
            
            elif score < threshold and classifier == 'True':
                 
                FN += 1
            
            elif score < threshold and classifier == 'False':
                
                TN += 1
        
        TPR = TP / (TP + FN) if TP + FN > 0 else 0
        TPRList.append(TPR)
        
        FPR = FP / (FP + TN) if FP + TN > 0 else 0
        FPRList.append(FPR)
        
        if TPR > 0.85 and TPR < 0.9:
            if FPR > 0.04 and FPR < 0.08:
                print(threshold)
                
    chartDataFrame = pd.DataFrame({
        "True Positive Rate": TPRList,
        "False Positive Rate": FPRList
    })
    
    return chartDataFrame

def create_classifier(scoreDataFrame):
    
    filenameList = scoreDataFrame['Fingerprint']
    scoreList = scoreDataFrame['Score']
    
    truePositiveList = []
    trueNegativeList = []
    classifierList = []
    
    for filename, score in zip(filenameList, scoreList):
        
        prefix, suffix = filename.split('_')
        
        if prefix == '101':
            
            classifierList.append((filename, 'True', score))

        else:
            
            classifierList.append((filename, 'False', score))
        
        classifierDataFrame = pd.DataFrame(classifierList, columns = \
            ['Fingerprint', 'Classifier', 'Score'])
        
        classifierDataFrame.sort_values('Fingerprint', inplace = True)
    
    return classifierDataFrame
            
def main():
    
    st.title('Fingerprint Analysis Tool')
    
    # File path for target fingerprints to be analysed by the user
    directoryPath = './target_fingerprint'
    
    # Initialise fingerprint database
    localStructsList = fp.analyse_fingerprints(None)

    # Uploade File button
    uploadedFile = st.file_uploader("Upload fingerprint image...", \
        type = ["png", "jpg", "jpeg", "tif"])

    if uploadedFile is not None:
        
        filePath = os.path.join(directoryPath, uploadedFile.name)
        
        with open(filePath, "wb") as f:
            
            f.write(uploadedFile.getbuffer())
        
        st.success('File Successfuly Uploaded')
        ls1 = fp.analyse_fingerprints(directoryPath)
        score = compare_fingerprints(localStructsList, ls1)
        st.write("Comparison Scores: ")
        st.dataframe(score, hide_index = True)
        
        classifier = create_classifier(score)
        st.write("Classifiers Table")
        st.dataframe(classifier, hide_index = True)
        
        ROCcurve = create_ROC_curve(classifier)
        st.line_chart(ROCcurve, x = "False Positive Rate", y = "True Positive Rate")

        # move the uploaded file to the fingerprint database directory
        newFilePath = os.path.join('./DB1_B - Copy', uploadedFile.name)
        os.rename(filePath, newFilePath)
      
if __name__ == "__main__":
    main()
    