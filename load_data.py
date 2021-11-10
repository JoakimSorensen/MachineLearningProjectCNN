"""
Created by Joakim Sorensen
2017008298
Machine Learning 2017
Kyung Hee University

NOTE: This script is heavily based on kjpark79's 
			script, presented on this blog:
			http://blog.naver.com/kjpark79/220783765651
			Most of his code is used directly or
			with  bit of modification.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf

filenames = []
batch_index = 0 
labels = []

"""
Get and append all the filenames to the filenames
array and shuffle the result. 
directory: the parent directory for the files
"""
def get_filenames(directory):
	global filenames, labels
	labels = []
	filenames = []

	for (dirpath, dirnames, filenames) in os.walk(directory):
		labels.extend(dirnames)
		break
	
	for i, label in enumerate(labels):
		list_dir = os.listdir(directory + label)
		for filename in list_dir:
			filenames.append([label + '/' + filename, i])

	random.shuffle(filenames)


"""
Get the number of files from the training folder.
returns the number of files for training
"""
def get_num_train_files():
	num_files = 0
	for (dirpath, dirnames, files) in os.walk('MathExprJpeg/train_data/'):
		if not dirpath == 'MathExprJpeg/train_data/':
			num_files += len(files)
		else:
			print('Not in parent directory: ', dirpath)
	return num_files


"""
Get the number of files from the testing folder.
returns the number of files for testing
"""
def get_num_test_files():
	num_files = 0
	for (dirpath, dirnames, files) in os.walk('MathExprJpeg/test_data/'):
		if not dirpath == 'MathExprJpeg/test_data/':
			num_files += len(files)
		else:
			print('Not in parent directory: ', dirpath)
	return num_files


"""
Get all the files from batch_index to batch_index + batch_size
and converts them to a numpy array/matrix.
sess: tensorflow session
batch_size: the batch size
num_classes: the number of classes
train: whether the training set should be used or not
returns the image data and the corresponsing one hot matrix as a tuple
"""
def get_jpeg_data(sess, batch_size, num_classes, train=True):
	global filenames, batch_index
	if train:	
		directory = 'MathExprJpeg/prototype_data/'
	else:
		directory = 'MathExprJpeg/test_directory/'
		
	# get the filenames if not done already
	if len(filenames) == 0:
		get_filenames(directory)

	max_ind = len(filenames)
	begin_ind = batch_index
	end_ind = batch_index + batch_size

	# if out of scope, start over
	if end_ind >= max_ind:
		end_ind = max_ind
		batch_index = 0
	
	x_data = np.array([])
	# a list of zeros for one hot encoding
	y_data = np.zeros((batch_size, num_classes))
	index = 0
	for i in range(begin_ind, end_ind):
		with tf.gfile.FastGFile(directory + filenames[i][0], 'rb') as f:
			#print('DirPath: ', directory + filenames[i][0])
			image_data = f.read()

		decode_image = tf.image.decode_jpeg(image_data, channels=1)
		# resixe image here if needed
		#-- resize --
		image = sess.run(decode_image)
		# append as array and divide with 255 to normalize
		x_data = np.append(x_data, np.asarray(image.data, dtype='float32')/255)

		y_data[index][filenames[i][1]] = 1
		index += 1

		# uncomment to see recreated image, used for debugging
		"""
		im = np.reshape(image.data, (45, 45, 1))
		print(y_data[index][filenames[i][1]])
		plt.imshow(im)
		plt.show()
		"""

	batch_index += batch_size

	try:
		# make it 2D
		x_data = x_data.reshape(batch_size, 45 * 45 * 1)
	except:
		print('ERROR')
		return None, None
	return x_data, y_data


"""
Get all the files from the selected dataset 
and converts them to a numpy array/matrix.
Saves the result in npy files, one for the image data,
one for the one hot labels ad one for the string labels.
sess: tensorflow session
num_classes: the number of classes
savefile: whether the datasets should be saved to files or not
trainset: whether the training set should be used or not
returns the image data and the corresponsing one hot matrix as a tuple
"""
def get_all_data(sess, num_classes, savefile=False, trainset=True):
	global filenames, labels

	if trainset:
		directory = 'MathExprJpeg/train_data/'
	else:
		directory = 'MathExprJpeg/test_data/'
		
	
	get_filenames(directory)

	max_ind = len(filenames)
	print('max_ind: ', max_ind)
	x_data = np.array([])
	# a list of zeros for one hot encoding
	y_data = np.zeros((max_ind, num_classes))
	index = 0
	for i in range(0, max_ind):
		with tf.gfile.FastGFile(directory + filenames[i][0], 'rb') as f:
			image_data = f.read()

		decode_image = tf.image.decode_jpeg(image_data, channels=1)
		image = sess.run(decode_image)
		# append as array and divide with 255 to normalize
		x_data = np.append(x_data, np.asarray(image.data, dtype='float32')/255)

		y_data[index][filenames[i][1]] = 1
		index += 1
		print("Reading test files: {:0.1f}%".format(100 * (i/max_ind)))
			
	try:
		# make it 2D
		x_data = x_data.reshape(max_ind, 45 * 45 * 1)
		if(savefile):
			np.save('x_data.npy', x_data)
			np.save('y_data.npy', y_data)
			np.save('labels.npy', labels)
	except:
		print('ERROR')
		return None, None
	return x_data, y_data
