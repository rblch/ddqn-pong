# preprocess/preprocess.py

import cv2
import numpy as np
from config.config import FRAME_SHAPE

class Preprocess:
    def __init__(self, frame_shape=FRAME_SHAPE):
        self.frame_h = frame_shape[0]
        self.frame_w = frame_shape[1]

    def preprocess(self, observation):
        # Convert to grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)

        # Drop top 20 pixels
        cropped = gray[20:, :]

        # Resize
        resized = cv2.resize(cropped, (self.frame_w, self.frame_h), interpolation=cv2.INTER_AREA)

        # Normalize 
        normalized = resized.astype(np.float32) / 255.0

        return normalized
