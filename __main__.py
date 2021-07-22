import handdetector as hd
import theremin
import cv2
import time
import sounddevice as sd
import tensorflow as tf
import ddsp
import ddsp.training
import os

# Load DDSP model
model_dir = 'model'
ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
ckpt_name = ckpt_files[0].split('.')[0]
ckpt = os.path.join(model_dir, ckpt_name)
model = ddsp.training.models.Autoencoder()
model.restore(ckpt)

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
success, img = cap.read()
h, w, c = img.shape
detector = hd.HandDetector(window_shape=(h, w))
tm = theremin.Theremin(detector=detector, model=model)

with sd.OutputStream(blocksize=tm.blocksize,
                     channels=tm.channels,
                     callback=tm.callback,
                     samplerate=tm.samplerate):
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        h, w, c = img.shape

        detector.find_hands(img, draw=True)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 255, 0), 3)

        cv2.imshow("Image", img)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
