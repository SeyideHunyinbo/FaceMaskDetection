import cv2
import numpy as np
from tensorflow.keras import models

model = models.load_model('facemaskdetectormodel2.h5')

results = {0 : 'without_mask', 1 : 'with_mask'}
dict_ = {0:(0,0,255), 1:(0,255,0)}

rect_size = 4
capture = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 
while True:
  rval, img = capture.read()
  img = cv2.flip(img, 1, 1)
  img_size = cv2.resize(img, (img.shape[1] // rect_size, img.shape[1] // rect_size))
  # faces = face_cascade.detectMultiScale(img_size)
  faces = face_cascade.detectMultiScale(img_size)

  for f in faces:
    (x, y, w, h) = [v * rect_size for v in f]

    face_img = img[y:y+h, x:x+w]
    img_resized = cv2.resize(face_img, (150, 150))
    normalized_img = img_resized / 255
    img_reshaped = np.reshape(normalized_img, (1, 150, 150, 3))
    img_reshaped = np.vstack([img_reshaped])
    result = model.predict(img_reshaped)

    label = np.argmax(result, axis=1)[0]

    cv2.rectangle(img, (x, y), (x+w, y+h), dict_[label], 2)
    cv2.rectangle(img, (x, y-40), (x+w, y), dict_[label], -1)
    cv2.putText(img, results[label], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

  cv2.imshow('LIVE', img)
  key = cv2.waitKey(10)

  if key == 27:
    break

capture.release()
cv2.destroyAllWindows()