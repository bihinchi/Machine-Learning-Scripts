from tensorflow.keras.preprocessing.image import random_rotation, random_shift, save_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import cv2, math, os
from random import random

import matplotlib.pyplot as plt
from data_structures import TrainingSequence



def get_sequence(data_path, batch_size=15, picture_size=(300,300), validation_split=False,
								   test=False,
								   rand_aug=True,
								   adv_prop=True,
								   noisy=False,
								   model=None,
								   changer=None):
											
	
	data_folder = "data_files"
	if not os.path.isdir(data_folder):
		os.mkdir(data_folder)
		
	
	X = []
	y = []
	no_label = []


	for index, foldeName in enumerate(sorted(os.listdir(data_path))):
		folderPath = os.path.join(data_path, foldeName)
		for i, image in enumerate(sorted(os.listdir(folderPath))):
			
			if not image.endswith("jpg"):
				continue
				
			if noisy:
				if random() < 0.3:
					no_label.append(os.path.join(folderPath, image))
				else:
					X.append(os.path.join(folderPath, image))
					y.append(index)
			
			else:
				X.append(os.path.join(folderPath, image))
				y.append(index)
			

	X = np.array(X)
	y = np.array(to_categorical(y, num_classes=2))
	
	if noisy:
		np.save(os.path.join(data_folder,"X"), X)
		np.save(os.path.join(data_folder,"y"), y)
		np.save(os.path.join(data_folder,"no_label"), no_label)
	

	if test:
		return TrainingSequence(X, batch_size, picture_size)

	if not validation_split:
		return TrainingSequence(X, y, batch_size, picture_size)

	if 0 < validation_split < 1:
		x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, train_size=1-validation_split,
															stratify=y)
															
		training_seq = TrainingSequence(x_train, y_train, batch_size, picture_size)
		validation_seq = TrainingSequence(x_test, y_test, batch_size, picture_size)
		return training_seq, validation_seq

