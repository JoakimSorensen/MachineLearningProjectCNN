"""
Created by Joakim Sorensen
2017008298
Machine Learning 2017
Kyung Hee University

"""

import os
import shutil
import random

"""
This script creates a train_data directory
and a test_data directory and puts specified
amount of training/testing files in them.
They are selected randomly.
"""
# original directory
org_directory = 'MathExprJpeg/extracted_images/'

train_directory = 'MathExprJpeg/train_data/'
test_directory = 'MathExprJpeg/test_data/'

os.makedirs(train_directory)
os.makedirs(test_directory)

# find the labels and append
labels = []
for(dirpath, dirnames, filenames) in os.walk(org_directory):
	labels.extend(dirnames)

for label in labels:
	os.makedirs(train_directory + label)
	os.makedirs(test_directory + label)

k = 0
desired_amount_of_train_files = 60
desired_amount_of_test_files = 20

for(dirpath, dirnames, filenames) in os.walk(org_directory):
	# take 20, or all, to be prototype data
	if not dirpath == org_directory:
		train_amount = desired_amount_of_train_files
		test_amount = desired_amount_of_test_files
		while (len(filenames) - train_amount) < 0:
			train_amount -= 1
		while (len(filenames) - test_amount) < 0:
			test_amount -= 1

		# to reduce the risk of the same file being picked two times
		random.shuffle(filenames)
		for i in range(train_amount):
			print(len(filenames))
			print(dirpath)
			shutil.copy(dirpath + '/' + filenames[i], train_directory + labels[k])

		random.shuffle(filenames) # get random files
		for i in range(test_amount):
			print(len(filenames))
			print(dirpath)
			shutil.copy(dirpath + '/' + filenames[i], test_directory + labels[k])
		k += 1
