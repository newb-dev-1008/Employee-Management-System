from netSystem import FaceRecognitionSystem

# -------------------------------- Script to train the recognizer --------------------------------
dataPath = os.path.join(mainDirName, "Datasets")
delete_Dataset = True
system_1 = FaceRecognitionSystem(datasetPathURL = dataPath, deleteDataset = delete_Dataset)
system_1.trainRecognizer()
# ------------------------------------------------------------------------------------------------