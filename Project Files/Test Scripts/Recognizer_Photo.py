import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from datetime import datetime

# Important paths for models and datasets
protoPathFolder = ""
modelPathFolder = ""
embedderPath = ""
recognizer_savePath = ""
labelencoder_savePath = ""

# Initialize face detector
protoPath = os.path.sep.join([protoPathFolder, 'deploy.prototxt'])
modelPath = os.path.sep.join([modelPathFolder, 'res10_300x300_ssd_iter_140000.caffemodel'])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load our serialized face embedding model
embedder = cv2.dnn.readNetFromTorch(embedderPath)

# Load the SVM Model and LabelEncoder
recognizer = pickle.loads(open(recognizer_savePath, "rb").read())
le = pickle.loads(open(labelencoder_savePath, "rb").read())

imageURL = "" # Image from S3 bucket (source yet to be known)
frame = downloadFrom(imageURL)
frame = imutils.resize(frame, width = 600)
(h, w) = frame.shape[:2]

imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177, 123.0), swapRB = False, crop = False)
detector.setInput(imageBlob)
    
detections = detector.forward()
    
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if (confidence > 0.2):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        face = frame[startY: endY, startX: endX]
        (fH, fW) = face.shape[:2]
        if ((fW < 20) or (fH < 20)):
            continue
        faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB = True, crop = False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()
        # print(vec)
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        returnStatus(None)
    else:
        returnStatus("Low confidence. Send a better photo.\n")
    
    
