import handdetector as hd
import theremin
import cv2
import time
import sounddevice as sd

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
success, img = cap.read()
h, w, c = img.shape
detector = hd.HandDetector(window_shape=(h, w))
tm = theremin.Theremin(detector=detector)

with sd.OutputStream(blocksize=tm.blocksize,
                     channels=tm.channels,
                     callback=tm.callback,
                     samplerate=tm.samplerate):
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        h, w, c = img.shape

        detector.find_hands(img, draw=True)
        #detector.update_center(img, 0)
        #detector.update_center(img, 1)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 255, 0), 3)

        cv2.imshow("Image", img)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
