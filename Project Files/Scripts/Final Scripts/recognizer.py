from netSystem import FaceRecognitionSystem

# -------------------------------- Script to recognize people real time --------------------------------
imageURL_1 = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Delete - Test images\Vikas\IMG-20190925-WA0008.jpg"

# Vikas
system_1 = FaceRecognitionSystem()
system_1.recognizePerson(imageURL_1)

# Vrushali
imageURL_2 = r"C:\Users\Yash Umale\Documents\7th Sem\Face Recognition Project\Delete - Test images\Vrushali\IMG-20210207-WA0151.jpg"
system_1.recognizePerson(imageURL_2)
# ------------------------------------------------------------------------------------------------------