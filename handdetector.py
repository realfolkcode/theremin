import cv2
import mediapipe as mp
import numpy as np


class HandDetector():
    def __init__(self, window_shape):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=2)
        self.mpDraw = mp.solutions.drawing_utils
        self.center = {"pitch": (0, 0),
                       "dynamics": (0, 0)}
        self.height = window_shape[0]
        self.width = window_shape[1]

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB.flags.writeable = False
        self.results = self.hands.process(imgRGB)
        imgRGB.flags.writeable = True

        h, w, c = img.shape
        self.height, self.width = h, w

        center = [self.center["pitch"], self.center["dynamics"]]
        i = 0
        if self.results.multi_hand_landmarks:
            print(len(self.results.multi_hand_landmarks))
            for hand_lms in self.results.multi_hand_landmarks:
                center[i] = self.get_center(img, hand_lms)
                i += 1
                #if i == 2:
                #    break
        if center[0][0] > center[1][0]:
            center[0], center[1] = center[1], center[0]
        self.center["pitch"] = center[0]
        self.center["dynamics"] = center[1]

        if draw:
            cv2.circle(img, (self.center["pitch"][0],
                             self.center["pitch"][1]),
                       radius=20,
                       color=(0, 0, 255),
                       thickness=cv2.FILLED)
            cv2.circle(img, (self.center["dynamics"][0],
                             self.center["dynamics"][1]),
                       radius=20,
                       color=(255, 0, 0),
                       thickness=cv2.FILLED)
        return img

    def find_position(self, img, hand_lms):
        lm_list = []
        h, w, c = img.shape
        for id, lm in enumerate(hand_lms.landmark):
            cx, cy = lm.x * w, lm.y * h
            lm_list.append([cx, cy])
        return np.array(lm_list)

    def get_center(self, img, hand_lms):
        lm_list = self.find_position(img, hand_lms)
        if len(lm_list) > 0:
            center = np.mean(lm_list, axis=0)
        return (int(center[0]), int(center[1]))