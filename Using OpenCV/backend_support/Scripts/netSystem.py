# This file handles the entire process of training and recognizing people real-time.
# Use trainer.py for specific operations that use methods defined here. 

# ---------------------------- Required Packages ----------------------------
from imutils import paths
import numpy as np
import imutils
import pickle
import time
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import argparse
import shutil
# ----------------------------------------------------------------------------

class FaceRecognitionSystem:

    # Set these paths as class variables if you do not wish the client company to set these. The paths you enter here 
    # would be treated as default for the system.
    # The paths passed by the client company will overwrite your default paths.

    mainDirName = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    datasetPathURL = os.path.join(mainDirName, "Datasets")
    protoPath = os.path.join(mainDirName, r"backend_support\Models\Face Detection\deploy.prototxt")
    modelPath = os.path.join(mainDirName, r"backend_support\Models\Face Detection\res10_300x300_ssd_iter_140000.caffemodel")
    embedderPath = os.path.join(mainDirName, r"backend_support\Models\Embedder\openface_nn4.small2.v1.t7")
    embeddingsFolder = os.path.join(mainDirName, r"backend_support\Embeddings and labels\Individual Embeddings")
    mainEmbeddingsPath = os.path.join(mainDirName, r"backend_support\Embeddings and labels\Collective\embeddings.pickle")
    savePathRecognizer = os.path.join(mainDirName, r"backend_support\Recognizer\recognizer.pickle") 
    savePathLabels = os.path.join(mainDirName, r"backend_support\Recognizer\le.pickle")

    def __init__(self, 
        datasetPathURL = datasetPathURL, 
        protoPath = protoPath, 
        modelPath = modelPath, 
        embedderPath = embedderPath, 
        embeddingsFolder = embeddingsFolder, 
        mainEmbeddingsPath = mainEmbeddingsPath, 
        savePathRecognizer = savePathRecognizer, 
        savePathLabels = savePathLabels,
        deleteDataset = True):

        self.datasetPathURL = datasetPathURL
        self.protoPathFolder = protoPath
        self.modelPathFolder = modelPath
        self.embedderPath = embedderPath
        self.embeddingsFolder = embeddingsFolder
        self.mainEmbeddingsPath = mainEmbeddingsPath
        self.savePathRecognizer = savePathRecognizer
        self.savePathLabels = savePathLabels
        self.deleteDataset = deleteDataset

    # --------------------------------------- Program to train an SVM recognizer ---------------------------------------

    def trainRecognizer(self):

        # Find number of people in the dataset
        numSaving = len(os.listdir(self.datasetPathURL))

        # If the dataset is empty, end the program here
        if (numSaving == 0):
            return "No data to train.\n"
        
        # Get names for all individuals (all folder names)
        names = os.listdir(self.datasetPathURL)

        # Initialize the face detector model
        detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)

        # Face Recognizer CNN - responsible for conversion of an image to a 128-D representation
        embedder = cv2.dnn.readNetFromTorch(self.embedderPath)

        # Generating 128-D representation embeddings for each person
        for name in names:
            userDatasetFolder = self.datasetPathURL + "\\" + name
            imagePaths = list(paths.list_images(userDatasetFolder))
            embedsFolder = self.embeddingsFolder + "\\" + name
            os.mkdir(embedsFolder)
            embeddingsPath = embedsFolder + "\\embeddings.pickle"

            userEmbeddings = []
            mainEmbeddings = []
            mainNames = []
            
            # If there are embeddings from a previous usage already present
            # And you're just adding a new person to an already existent embeddings storage location
            if (len(os.listdir(self.embeddingsFolder)) != 0):
                
                # Load the embeddings of the earlier people into the lists
                # To include them collectively in the new training process
                data = pickle.loads(open(self.mainEmbeddingsPath, "rb").read())
                mainNames = data["Names"]
                mainEmbeddings = data["Embeddings"]


            for (i, imagePath) in enumerate(imagePaths):
                name = imagePath.split(os.path.sep)[-2]
                image = cv2.imread(imagePath)
                image = imutils.resize(image, width = 600)
                (h, w) = image.shape[:2]                                                        
                imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
                detector.setInput(imageBlob)                                                    
                detections = detector.forward()

                # If any face is detected in the image
                if (len(detections) > 0):                                                         
                    i = np.argmax(detections[0, 0, :, 2])
                    confidence = detections[0, 0, i, 2]

                    # If the model is at least 70% confident that it has found a face
                    # It'll create a face blob out of the image
                    if (confidence > 0.7):                                                          
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        face = image[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]	
                        if (fW < 20 or fH < 20):
                            continue
                        
                        # Get the 128-D output from the CNN
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)                   
                        embedder.setInput(faceBlob)
                        vec = embedder.forward()

                        # Add the new embedding to the list of embeddings 
                        mainEmbeddings.append(vec.flatten())
                        userEmbeddings.append(vec.flatten())
                        mainNames.append(name)

            # Write the updated embeddings list (with previous people + newly added person) back to file
            data = {"Embeddings" : mainEmbeddings, "Names" : mainNames}
            f = open(self.mainEmbeddingsPath, "wb")
            f.write(pickle.dumps(data))
            f.close()

            # Create a new embeddings pickle file for the specific person and all their images
            # Not necessary, but good to have instead of storing entire datasets
            userEmbeddingsPath = embeddingsPath
            f = open(userEmbeddingsPath, "wb")
            f.write(pickle.dumps(userEmbeddings))
            f.close()
        
            # Delete the current dataset folder and its photos
            # As training is done, in the interest of space
            if (self.deleteDataset == True):
                shutil.rmtree(userDatasetFolder)

        # Training the SVM on new labels and embeddings - only if the number of folders in embeddingsFolder is more than 2
        # One folder for collective embeddings of all people, another for any person whose photos have been trained.
        if (len(os.listdir(self.embeddingsFolder)) >= 2):
            data = pickle.loads(open(self.mainEmbeddingsPath, 'rb').read())
            le = LabelEncoder()
            labels = le.fit_transform(data['Names'])

            param_grid = {'C' : [0.001, 0.01, 0.1, 1, 10, 100], 'gamma' : [100, 10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel' : ['linear', 'rbf', 'poly']}

            grid = GridSearchCV(SVC(probability = True), param_grid, scoring = "precision", refit = True, verbose = 3, n_jobs = -1)
            grid.fit(data["Embeddings"], labels)

            f = open(self.savePathRecognizer, "wb")
            f.write(pickle.dumps(grid))
            f.close()

            # Save the label encoder
            f = open(self.savePathLabels, "wb")
            f.write(pickle.dumps(le))
            f.close()

    # ---------------------------------------------------------------------------------------------------------------------

    # --------------------------------------- Program to recognize people real-time ---------------------------------------

    def recognizePerson(self, imageURL):

        # Initialize face detector
        detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)

        # Load our serialized face embedding model
        embedder = cv2.dnn.readNetFromTorch(self.embedderPath)

        # Load the SVM Model and LabelEncoder
        recognizer = pickle.loads(open(self.savePathRecognizer, "rb").read())
        le = pickle.loads(open(self.savePathLabels, "rb").read())

        frame = cv2.imread(imageURL)
        frame = imutils.resize(frame, width = 600)
        (h, w) = frame.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177, 123.0), swapRB = False, crop = False)
        detector.setInput(imageBlob)
            
        detections = detector.forward()
            
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if (confidence > 0.6):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                face = frame[startY: endY, startX: endX]
                (fH, fW) = face.shape[:2]
                if ((fW < 20) or (fH < 20)):
                    continue
                faceBlob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), swapRB = True, crop = False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                preds = recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]

                if (proba > 0.7):
                    name = le.classes_[j]
                else:
                    name = "Unknown"
                    
                prediction = name
            
        os.remove(imageURL)
        return prediction

    # ------------------------------------------------------------------------------------------------------------------

    # --------------------------------------- Function to perform face alignment ---------------------------------------
    def faceAlignment():
        return
    # ------------------------------------------------------------------------------------------------------------------

    # -------------------------------------- Function to perform data augmentation -------------------------------------
    def dataAugmentation():
        return
    # ------------------------------------------------------------------------------------------------------------------ 