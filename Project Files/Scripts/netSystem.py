# This file handles the entire process of training and recognizing people real-time.
# The methods defined in this class are executed sequentially as displayed under #main 

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
import argparse
import shutil
# ----------------------------------------------------------------------------

class FaceRecognitionSystem:

    # Set these paths as class variables if you do not wish the client company to set these. The paths you enter here 
    # would be treated as default for the system.
    # The paths passed by the client company will overwrite your default paths.

    datasetPathURL = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Datasets"
    protoPathFolder = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Models\Face Detection"
    modelPathFolder = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Models\Face Detection"
    embedderPath = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Models\Embedder\openface_nn4.small2.v1.t7"
    embeddingsFolder = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Embeddings and labels\Individual Embeddings"
    mainEmbeddingsPath = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Embeddings and labels\Collective\embeddings.pickle"
    savePathRecognizer = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Recognizer\recognizer.pickle" 
    savePathLabels = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Recognizer\le.pickle"

    def __init__(self, datasetPathURL = datasetPathURL, protoPathFolder = protoPathFolder, modelPathFolder = modelPathFolder, 
    embedderPath = embedderPath, embeddingsFolder = embeddingsFolder, mainEmbeddingsPath = mainEmbeddingsPath, 
    savePathRecognizer = savePathRecognizer, savePathLabels = savePathLabels):

        self.datasetPathURL = datasetPathURL
        self.protoPathFolder = protoPathFolder
        self.modelPathFolder = modelPathFolder
        self.embedderPath = embedderPath
        self.embeddingsFolder = embeddingsFolder
        self.mainEmbeddingsPath = mainEmbeddingsPath
        self.savePathRecognizer = savePathRecognizer
        self.savePathLabels = savePathLabels

    def trainRecognizer(self):
        # Argparse arguments
        savingMultiple = str(input("Saving multiple people?: ")) 
        if (savingMultiple == "No"):
            name = # Name of person
        else:
            names = # List of names in order of photos

        # --------------------------------------- Main Program ---------------------------------------

        # FaceNet
        protoPath = os.path.sep.join([self.protoPathFolder, 'deploy.prototxt'])
        modelPath = os.path.sep.join([self.modelPathFolder, 'res10_300x300_ssd_iter_140000.caffemodel'])
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # Face Recognizer
        embedder = cv2.dnn.readNetFromTorch(self.embedderPath)

        if (savingMultiple == "No"): 
            userDatasetFolder = self.datasetFolder + "\\" + name # Folder containing images of a specific user

            if (len(os.listdir(self.embeddingsFolder)) == 0):
                imagePaths = list(paths.list_images(userDatasetFolder))

                embedsFolder = self.embeddingsFolder + "\\" + name
                os.mkdir(embedsFolder)
                embeddingsPath = embedsFolder + "\\embeddings.pickle"

                knownEmbeddings = []
                mainEmbeddings = []
                mainNames = []

                for (i, imagePath) in enumerate(imagePaths):
                    name = imagePath.split(os.path.sep)[-2]
                    image = cv2.imread(imagePath)
                    image = imutils.resize(image, width = 600)

                    (h, w) = image.shape[:2]                                                        
                    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
                    detector.setInput(imageBlob)                                                    
                    detections = detector.forward()

                    if (len(detections) > 0):                                                         
                        i = np.argmax(detections[0, 0, :, 2])
                        confidence = detections[0, 0, i, 2]  

                        if (confidence > 0.1):                                                          
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            face = image[startY:endY, startX:endX]
                            (fH, fW) = face.shape[:2]	

                            if (fW < 20 or fH < 20):
                                continue    

                            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)                   
                            embedder.setInput(faceBlob)
                            vec = embedder.forward()

                            mainEmbeddings.append(vec.flatten())
                            knownEmbeddings.append(vec.flatten())
                            mainNames.append(name)

                data = {"Embeddings" : mainEmbeddings, "Names" : mainNames}
                f = open(self.mainEmbeddingsPath, "wb")
                f.write(pickle.dumps(data))
                f.close()

                userEmbeddingsPath = embeddingsPath
                f = open(userEmbeddingsPath, "wb")
                f.write(pickle.dumps(knownEmbeddings))
                f.close()

            else:
                imagePaths = list(paths.list_images(userDatasetFolder))
                embedsFolder = self.embeddingsFolder + "\\" + name
                os.mkdir(embedsFolder)
                embeddingsPath = embedsFolder + "\\embeddings.pickle"

                data = pickle.loads(open(self.mainEmbeddingsPath, "rb").read())
                knownEmbeddings = []
                previouslyKnownEmbeddings = data["Embeddings"]
                previouslyKnownNames = data["Names"]

                for (i, imagePath) in enumerate(imagePaths):
                    name = imagePath.split(os.path.sep)[-2]
                    image = cv2.imread(imagePath)
                    image = imutils.resize(image, width = 600)

                    (h, w) = image.shape[:2]                                                        
                    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
                    detector.setInput(imageBlob)                                                    
                    detections = detector.forward()

                    if (len(detections) > 0):                                                         
                        i = np.argmax(detections[0, 0, :, 2])
                        confidence = detections[0, 0, i, 2]  

                        if (confidence > 0.1):                                                          
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            face = image[startY:endY, startX:endX]
                            (fH, fW) = face.shape[:2]	

                            if (fW < 20 or fH < 20):
                                continue    

                            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)                   
                            embedder.setInput(faceBlob)
                            vec = embedder.forward()

                            previouslyKnownNames.append(name)                                                    
                            previouslyKnownEmbeddings.append(vec.flatten())
                            knownEmbeddings.append(vec.flatten())

                data = {"Embeddings" : previouslyKnownEmbeddings, "Names" : previouslyKnownNames}
                f = open(self.mainEmbeddingsPath, "wb")
                f.write(pickle.dumps(data))
                f.close()

                userEmbeddingsPath = embeddingsPath
                f = open(userEmbeddingsPath, "wb")
                f.write(pickle.dumps(knownEmbeddings))
                f.close()
            

            shutil.rmtree(userDatasetFolder)

        else:
            for name in names:
                userDatasetFolder = self.datasetFolder + "\\" + name
                
                if (len(os.listdir(self.embeddingsFolder)) == 0):
                    imagePaths = list(paths.list_images(userDatasetFolder))
                    embedsFolder = self.embeddingsFolder + "\\" + name
                    os.mkdir(embedsFolder)
                    embeddingsPath = embedsFolder + "\\embeddings.pickle"
                    
                    knownEmbeddings = []
                    mainEmbeddings = []
                    mainNames = []

                    for (i, imagePath) in enumerate(imagePaths):
                        name = imagePath.split(os.path.sep)[-2]
                        image = cv2.imread(imagePath)
                        image = imutils.resize(image, width = 600)

                        (h, w) = image.shape[:2]                                                        
                        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
                        detector.setInput(imageBlob)                                                    
                        detections = detector.forward()

                        if (len(detections) > 0):                                                         
                            i = np.argmax(detections[0, 0, :, 2])
                            confidence = detections[0, 0, i, 2]  

                            if (confidence > 0.1):                                                          
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")

                                face = image[startY:endY, startX:endX]
                                (fH, fW) = face.shape[:2]	

                                if (fW < 20 or fH < 20):
                                    continue    

                                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)                   
                                embedder.setInput(faceBlob)
                                vec = embedder.forward()

                                mainEmbeddings.append(vec.flatten())
                                knownEmbeddings.append(vec.flatten())
                                mainNames.append(name)

                    data = {"Embeddings" : mainEmbeddings, "Names" : mainNames}
                    f = open(self.mainEmbeddingsPath, "wb")
                    f.write(pickle.dumps(data))
                    f.close()

                    userEmbeddingsPath = embeddingsPath
                    f = open(userEmbeddingsPath, "wb")
                    f.write(pickle.dumps(knownEmbeddings))
                    f.close()
                
                else:
                    imagePaths = list(paths.list_images(userDatasetFolder))
                    embedsFolder = self.embeddingsFolder + "\\" + name
                    os.mkdir(embedsFolder)
                    embeddingsPath = embedsFolder + "\\embeddings.pickle"

                    data = pickle.loads(open(self.mainEmbeddingsPath, "rb").read())
                    knownEmbeddings = []
                    previouslyKnownEmbeddings = data["Embeddings"]
                    previouslyKnownNames = data["Names"]

                    for (i, imagePath) in enumerate(imagePaths):
                        name = imagePath.split(os.path.sep)[-2]
                        image = cv2.imread(imagePath)
                        image = imutils.resize(image, width = 600)

                        (h, w) = image.shape[:2]                                                        
                        imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
                        detector.setInput(imageBlob)                                                    
                        detections = detector.forward()

                        if (len(detections) > 0):                                                         
                            i = np.argmax(detections[0, 0, :, 2])
                            confidence = detections[0, 0, i, 2]  

                            if (confidence > 0.1):                                                          
                                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                (startX, startY, endX, endY) = box.astype("int")

                                face = image[startY:endY, startX:endX]
                                (fH, fW) = face.shape[:2]	

                                if (fW < 20 or fH < 20):
                                    continue    

                                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)                   
                                embedder.setInput(faceBlob)
                                vec = embedder.forward()

                                previouslyKnownNames.append(name)                                                    
                                previouslyKnownEmbeddings.append(vec.flatten())
                                knownEmbeddings.append(vec.flatten())

                        data = {"Embeddings" : previouslyKnownEmbeddings, "Names" : previouslyKnownNames}
                        f = open(self.mainEmbeddingsPath, "wb")
                        f.write(pickle.dumps(data))
                        f.close()

                        userEmbeddingsPath = embeddingsPath
                        f = open(userEmbeddingsPath, "wb")
                        f.write(pickle.dumps(knownEmbeddings))
                        f.close()
            

            shutil.rmtree(userDatasetFolder)

        # Training the SVM on new labels and embeddings
        if (len(os.listdir(self.embeddingsFolder)) >= 2):
            data = pickle.loads(open(self.mainEmbeddingsPath, 'rb').read())
            le = LabelEncoder()
            labels = le.fit_transform(data['Names'])

            recognizer = SVC(C = 1.0, kernel = "linear", probability = True)
            recognizer.fit(data["Embeddings"], labels)

            f = open(self.savePathRecognizer, "wb")
            f.write(pickle.dumps(recognizer))
            f.close()

            # Save the label encoder
            f = open(self.savePathLabels, "wb")
            f.write(pickle.dumps(le))
            f.close()
