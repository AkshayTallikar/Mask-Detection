import cv2
import numpy as np
from keras.models import load_model

new_model = load_model("Resources/model-003.model")
frameWidth = 400
frameHeight = 400
face_cascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
color = (255, 0, 255)
cap = cv2.VideoCapture("Resources/IMG_4270.mp4")
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
labels_dict = {0: 'MASK', 1: 'NO MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgGray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgGray2, 1.1, 10)
    for (x, y, w, h) in faces:
        face_img = imgGray[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (120, 120))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 120, 120, 3))
        result = new_model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]
        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("video", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
