# preprocess/frame_stack.py

from collections import deque
import numpy as np
from config.config import NUM_FRAMES, FRAME_SHAPE

class FrameStack:
    def __init__(self, num_frames=NUM_FRAMES, frame_shape=FRAME_SHAPE):
        self.num_frames = num_frames
        self.frame_shape = frame_shape
        self.frames = deque(maxlen=num_frames)
    
    def push(self, frame):
        self.frames.append(frame)
    
    def get_stacked_frames(self):
        # If not enough frames, pad with zeros
        while len(self.frames) < self.num_frames:
            self.frames.append(np.zeros(self.frame_shape, dtype=np.float32))
        stacked = np.stack(self.frames, axis=0)
        return stacked
