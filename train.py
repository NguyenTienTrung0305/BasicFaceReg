import os
import numpy as np
import cv2
from PIL import Image # library for handle image 


recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "Project/FaceRecognizationV2/dataset"

def get_images_with_id(path):
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for image_path in images_path:
        face_img = Image.open(image_path).convert("L") # chuyển ảnh sang chế độ grayscale (ảnh xám)
        face_np = np.array(face_img, np.uint8) # chuyển ảnh thành mảng array từ 0-255
        id = int(os.path.split(image_path)[-1].split(".")[1]) # lấy id của user

        faces.append(face_np)
        ids.append(id)

        cv2.imshow("training", face_np) # imshow nhận tham số là mảng array
        cv2.waitKey(10)

    return np.array(ids), faces

ids, faces = get_images_with_id(path=path)

# train model
# recognizer.train() nhận 2 tham số 
    # + images: danh sách các ảnh xám (numpy array)
    # + labels: là danh sách các nhãn, mỗi nhãn là một số nguyên đại diện cho danh tính của người trong ảnh tương ứng.
recognizer.train(faces, ids) 

# save all faces be trained
recognizer.save("Project/FaceRecognizationV2/recognizer/trainingdata.yml")
cv2.destroyAllWindows()

