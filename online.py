from sklearn.preprocessing import LabelEncoder
import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from keras_facenet import FaceNet
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


facenet = FaceNet()
faces_embeddings = np.load("C:/proyek/face_recognition/faces_embeddings.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model.pkl", 'rb'))
# Ganti parameter menjadi 0 untuk menggunakan webcam bawaan
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    for (x, y, w, h) in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)
        embedding = facenet.embeddings(img)
        face_name = model.predict(embedding)
        final_name = encoder.inverse_transform(face_name)
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv.putText(frame, str(final_name), (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
