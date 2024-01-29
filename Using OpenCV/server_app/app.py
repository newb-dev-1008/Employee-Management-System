from flask import Flask, request
from waitress import serve
import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), r'backend_support\Scripts'))

from netSystem import FaceRecognitionSystem

# Check if the current system has dlib support.
# If it does, use dlib; else use general OpenCV techniques.
dlib_installed = True
try:
    import dlib
except ImportError:
    dlib_installed = False

PHOTO_PATH = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['PHOTO_PATH'] = PHOTO_PATH

@app.route('/', methods=['GET', 'POST'])
def recognizerListener():

    uploaded_files = request.files.getlist('file')
    recognized_people = []

    # Loop through uploaded files
    for file in uploaded_files:
        upload_path = os.path.join(app.config.get('PHOTO_PATH'), file.filename)
        file.save(upload_path)

        # Recognize person for each file
        system = FaceRecognitionSystem()
        recognized_person = system.recognizePerson(upload_path)
        # recognized_people.append(recognized_person)

    # Return list of recognized people
    return recognized_person
    
    # Uploading datasets based on requirement
    #for f in request.files.getlist('file'):
    #    uploadPath = os.path.join(app.config.get('PHOTO_PATH'), f.filename)
    #    f.save(uploadPath)

    ## -------------------------------- Script to recognize person --------------------------------
    # system_1 = FaceRecognitionSystem()
    # recognizedPerson = system_1.recognizePerson(uploadPath)

    # Post the returned type
    # return recognizedPerson
    # --------------------------------------------------------------------------------------------


if (__name__ == "__main__"):
    app.run(debug = True) # [Was valid for running a test server with Flask]
    # serve(app, host = '0.0.0.0', port = 5000)