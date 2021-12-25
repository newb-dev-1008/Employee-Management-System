from flask import Flask, request
from Scripts.netSystem import FaceRecognitionSystem
import os

UPLOAD_FOLDER_PATH = r"Datasets"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_PATH 

@app.route('/Test', methods=['POST'])
def uploadListener():

    # Uploading datasets based on requirement
    username = request.form.get('Name')
    os.mkdir(UPLOAD_FOLDER_PATH + '\\' + username)
    for f in request.files.getlist('file'):
        f.save(os.path.join(app.config.get('UPLOAD_FOLDER'), username, f.filename))

    # -------------------------------- Script to train the recognizer --------------------------------
    dataPath = r"C:\Users\Yash Umale\Documents\7th Sem\Side Projects\Face Recognition Project\Delete - Datasets\Test folder"
    system_1 = FaceRecognitionSystem(datasetPathURL = dataPath)
    system_1.trainRecognizer()
    # ------------------------------------------------------------------------------------------------
    
if (__name__ == "__main__"):
    app.run(debug =True)
