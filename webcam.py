from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
import time


def detector(frame, facenet, masknet):
    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    facenet.setInput(blob)
    detection = facenet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(detection.shape[2]):
        box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")
        start_x, start_y = (max(start_x, 0), max(start_y, 0))
        end_x, end_y = (min(w - 1, end_x), min(h - 1, end_y))

        face = frame[start_y:end_y, start_x:end_x]
        face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        face = cv.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        faces.append(face)
        locs.append((start_x, start_y, end_x, end_y))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = masknet.predict(faces, batch_size=32)

        return locs, preds


prototxt = "models//deploy.prototxt"
weights = "models//res10_300x300_ssd_iter_140000.caffemodel"
facenet = cv.dnn.readNet(prototxt, weights)

masknet = load_model("models//model.h5")

webcam_video = cv.VideoCapture(0)
time.sleep(2.0)
while webcam_video.isOpened():
    ret, frame = webcam_video.read()
    frame = cv.resize(frame, (600, 600))

    locs, preds = detector(frame, facenet, masknet)

    for box, pred in zip(locs, preds):
        s_x, s_y, e_x, e_y = box

        with_mask, without_mask = pred

        label = "Mask" if with_mask > without_mask else "NO Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{} : {}".format(label, max(with_mask, without_mask) * 100)

        cv.putText(frame, label, (s_x, s_y - 10),
                   cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.50, color, 2)

        cv.rectangle(frame, (s_x, s_y), (e_x, e_y), color, 2)

        cv.imshow("Face", frame)
        if cv.waitKey(1) == ord("q"):
            break

webcam_video.release()
cv.destroyAllWindows()
