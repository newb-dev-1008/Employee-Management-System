# Steps for training from Android app:
# 1. User takes photos of the person
# 2. User presses a button to push all photos to storage, along with person's name
# 3. Once in S3, the Python script is triggered using an API
# 4. In S3, each user has a seperate folder and each folder will have embeddings.pickle and le.pickle ("~root/Users/<Name>/...")
# 5. If the Users folder is empty, then create a new folder and create the pickle files but don't train the SVM. Also, delete the dataset folder per user
# 6. For subsequent new users, check if the Users folder has at least one user. If it does, then recreate the data dictionary and create pickle files. Proceed to training.
# 7. Save the new SVM pickle in Model folder.

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

# --------------------------------------- Variables and paths for configuration ---------------------------------------

# Important paths for models and datasets
protoPathFolder = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Models\Face Detection"
modelPathFolder = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Models\Face Detection"
embedderPath = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Models\Embedder\openface_nn4.small2.v1.t7"
embeddingsFolder = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Embeddings and labels\Individual Embeddings"
mainEmbeddingsPath = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Embeddings and labels\Collective\embeddings.pickle"
datasetFolder = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Datasets"

savePathRecognizer = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Recognizer\recognizer.pickle" 
savePathLabels = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Misc Files\Recognizer\le.pickle"


# Argparse arguments
savingMultiple = str(input("Saving multiple people?: ")) 
if (savingMultiple == "No"):
  name = # Name of person
else:
  names = # List of names in order of photos

# ---------------------------------------------------------------------------------------------------------------------

# --------------------------------------- Main Program ---------------------------------------

# FaceNet
protoPath = os.path.sep.join([protoPathFolder, 'deploy.prototxt'])
modelPath = os.path.sep.join([modelPathFolder, 'res10_300x300_ssd_iter_140000.caffemodel'])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Face Recognizer
embedder = cv2.dnn.readNetFromTorch(embedderPath)

if (savingMultiple == "No"): 
  userDatasetFolder = datasetFolder + "\\" + name # Folder containing images of a specific user

  if (len(os.listdir(embeddingsFolder)) == 0):
    imagePaths = list(paths.list_images(userDatasetFolder))

    embedsFolder = embeddingsFolder + "\\" + name
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
    f = open(mainEmbeddingsPath, "wb")
    f.write(pickle.dumps(data))
    f.close()

    userEmbeddingsPath = embeddingsPath
    f = open(userEmbeddingsPath, "wb")
    f.write(pickle.dumps(knownEmbeddings))
    f.close()

  else:
    imagePaths = list(paths.list_images(userDatasetFolder))
    # embeddingsPath = embeddingsFolder + "\\" + name + "\\embeddings.pickle"
    embedsFolder = embeddingsFolder + "\\" + name
    os.mkdir(embedsFolder)
    embeddingsPath = embedsFolder + "\\embeddings.pickle"

    data = pickle.loads(open(mainEmbeddingsPath, "rb").read())
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
    f = open(mainEmbeddingsPath, "wb")
    f.write(pickle.dumps(data))
    f.close()

    userEmbeddingsPath = embeddingsPath
    f = open(userEmbeddingsPath, "wb")
    f.write(pickle.dumps(knownEmbeddings))
    f.close()
  

  shutil.rmtree(userDatasetFolder)

else:
  for name in names:
    userDatasetFolder = datasetFolder + "\\" + name
    
    if (len(os.listdir(embeddingsFolder)) == 0):
      imagePaths = list(paths.list_images(userDatasetFolder))
      embedsFolder = embeddingsFolder + "\\" + name
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
      f = open(mainEmbeddingsPath, "wb")
      f.write(pickle.dumps(data))
      f.close()

      userEmbeddingsPath = embeddingsPath
      f = open(userEmbeddingsPath, "wb")
      f.write(pickle.dumps(knownEmbeddings))
      f.close()
    
    else:
      imagePaths = list(paths.list_images(userDatasetFolder))
      embedsFolder = embeddingsFolder + "\\" + name
      os.mkdir(embedsFolder)
      embeddingsPath = embedsFolder + "\\embeddings.pickle"

      data = pickle.loads(open(mainEmbeddingsPath, "rb").read())
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
        f = open(mainEmbeddingsPath, "wb")
        f.write(pickle.dumps(data))
        f.close()

        userEmbeddingsPath = embeddingsPath
        f = open(userEmbeddingsPath, "wb")
        f.write(pickle.dumps(knownEmbeddings))
        f.close()
  

    shutil.rmtree(userDatasetFolder)

# Training
if (len(os.listdir(embeddingsFolder)) >= 2):
  data = pickle.loads(open(mainEmbeddingsPath, 'rb').read())
  le = LabelEncoder()
  labels = le.fit_transform(data['Names'])

  print("Training SVM on ", len(data), " labels.\n")
  recognizer = SVC(C = 1.0, kernel = "linear", probability = True)
  recognizer.fit(data["Embeddings"], labels)

  f = open(savePathRecognizer, "wb")
  f.write(pickle.dumps(recognizer))
  f.close()

  # Save the label encoder
  f = open(savePathLabels, "wb")
  f.write(pickle.dumps(le))
  f.close()

# --------------------------------------------------------------------------------------------