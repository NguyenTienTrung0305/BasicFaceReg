import cv2
import numpy as np
import sqlite3
from PIL import Image

from scipy.datasets import face

face_detect = cv2.CascadeClassifier("Project/FaceRecognizationV2/haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

def insert_or_update(Id, name, age):
    conn = sqlite3.connect("Project/FaceRecognizationV2/sqlite.db")
    cmd = "SELECT * FROM STUDENTS WHERE ID=" + str(Id)
    cursor = conn.execute(cmd)

    is_already_exist = 0

    for row in cursor:
        is_already_exist = 1

    if is_already_exist == 1: # if already exist face in data => update
        conn.execute("UPDATE STUDENTS SET Name=? WHERE ID=?", (name,Id,))
        conn.execute("UPDATE STUDENTS SET Age=? WHERE ID=?", (age,Id,))
    else:
        conn.execute("INSERT INTO STUDENTS (ID,Name,Age) values (?,?,?)", (Id,name,age)) 

    
    conn.commit()
    conn.close()

Id = input("enter ID:")
name = input("enter name:")
age = input("enter age:")
insert_or_update(Id=Id, name=name, age=age)

sample_num = 0 # assume their's no samples in dataset

# detect face in webcam coding
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    for x,y,w,h in faces:
        sample_num+=1
        cv2.imwrite("Project/FaceRecognizationV2/dataset/user." + str(Id) + "." + str(sample_num) + ".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.waitKey(100) # delay time
    
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if sample_num > 20:
        break

# # receive data from image
# img = "Project/FaceRecognizationV2/test/ID2201.jpg"
# gray = Image.open(img).convert("L") # convert image to grayscale
# img_face = np.array(gray, np.uint8) # convert image to numpy array
# faces = face_detect.detectMultiScale(img_face, 1.3, 5)
# for x,y,w,h in faces:
#     face = img_face[y:y+h, x:x+w]
#     cv2.imwrite("Project/FaceRecognizationV2/dataset/user." + str(Id) + "." + str(sample_num) + ".jpg", face)
#     sample_num += 1
#     cv2.waitKey(100) # delay time



cam.release()
cv2.destroyAllWindows()