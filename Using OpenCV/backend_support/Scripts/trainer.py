from netSystem import FaceRecognitionSystem
import os

# Check if the current system has dlib support.
# If it does, use dlib; else use general OpenCV techniques.
dlib_installed = True
try:
    import dlib
except ImportError:
    dlib_installed = False
    print("dlib Package not found - implementing OpenCV approach.\n")

# -------------------------------- Script to train the recognizer --------------------------------
dataPath = os.path.join(FaceRecognitionSystem.mainDirName, "Datasets")

# Set this to False if you don't want your datasets to be deleted after training
delete_Dataset = False

system_1 = FaceRecognitionSystem(datasetPathURL = dataPath, deleteDataset = delete_Dataset)
system_1.trainRecognizer()
# ------------------------------------------------------------------------------------------------