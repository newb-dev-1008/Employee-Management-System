from netSystem import FaceRecognitionSystem
import urllib.request
import os

'''# --------------------------------- Script to recognize people real time --------------------------------

# TODO: Set directory where images can be downloaded for recognition and deleted immediately
temporary_directory = r""
os.chdir(temporary_directory)

# Adding information about user agent
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
urllib.request.install_opener(opener)

# TODO: Receive image URL 
imageURL = ""
filename = imageURL.split("/")[-1]

# Download file into filename.jpg
urllib.request.urlretrieve(imageURL, filename)

# Pass the downloaded file path into recognizePerson()
imagePath = os.path.sep.join(temporary_directory, filename)
system = FaceRecognitionSystem()
system.recognizePerson(imagePath)

# -------------------------------------------------------------------------------------------------------'''

# -------------------------------- Test script to recognize known people --------------------------------
imageURL_1 = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Delete - Test images\Vikas\IMG-20190925-WA0008.jpg"

# Vikas
system_1 = FaceRecognitionSystem()
print(system_1.recognizePerson(imageURL_1))

# Vrushali
imageURL_2 = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Delete - Test images\Vrushali\IMG-20210207-WA0151.jpg"
print(system_1.recognizePerson(imageURL_2))
# -------------------------------------------------------------------------------------------------------