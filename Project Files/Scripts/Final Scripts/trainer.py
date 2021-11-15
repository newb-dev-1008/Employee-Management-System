from netSystem import FaceRecognitionSystem

# -------------------------------- Script to train the recognizer --------------------------------
dataPath = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Delete - Datasets\Test folder"
system_1 = FaceRecognitionSystem(datasetPathURL = dataPath)
system_1.trainRecognizer()
# ------------------------------------------------------------------------------------------------