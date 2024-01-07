from netSystem import FaceRecognitionSystem

# -------------------------------- Script to train the recognizer --------------------------------
dataPath = r"C:\Users\Yash Umale\Downloads\Sayanth Stuff\vcz_images"
system_1 = FaceRecognitionSystem(datasetPathURL = dataPath)
system_1.trainRecognizer()
# ------------------------------------------------------------------------------------------------