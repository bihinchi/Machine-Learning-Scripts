from tensorflow.keras.utils import Sequence
import numpy as np
import cv2, math


class TrainingSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size=30, pic_size=(200,200)):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.pic_size = pic_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
        self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
        self.batch_size]

        return np.array([cv2.resize(cv2.imread(file_name), self.pic_size)
                    for file_name in batch_x]), np.array(batch_y)

