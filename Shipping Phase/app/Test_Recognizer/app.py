from flask import Flask, request
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), r'Scripts'))
from netSystem import FaceRecognitionSystem

PHOTO_PATH = r"C:\Users\Yash Umale\Documents\7th Sem\Side Projects\Face Recognition Project\Shipping Phase\app\Test_Recognizer"

app = Flask(__name__)
app.config['PHOTO_PATH'] = PHOTO_PATH

@app.route('/', methods=['GET', 'POST'])
def recognizerListener():
    
    # Uploading datasets based on requirement
    for f in request.files.getlist('file'):
        uploadPath = os.path.join(app.config.get('PHOTO_PATH'), f.filename)
        f.save(uploadPath)

    # -------------------------------- Script to recognize person --------------------------------
    system_1 = FaceRecognitionSystem()
    recognizedPerson = system_1.recognizePerson(uploadPath)

    # Post the returned type
    return recognizedPerson
    # --------------------------------------------------------------------------------------------

if (__name__ == "__main__"):
    app.run(debug = True)