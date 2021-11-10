"""
Created by Joakim Sorensen
2017008298
Machine Learning 2017
Kyung Hee University

"""

from cnn_model import Model
import load_data
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

"""
This is the main script for running either an ensemble of
the CNN model or a single model, set by ensemble_mode. 
In order to read from files the script create_datasets.py 
has to be run in advance. 

This script is set for 82 classes, this can however be changed
along with the dataset being used.

After training the model is tested with the test dataset right away.

Accuracy and precision will be displayed in the terminal
together with the predicted expression and the correct label.

"""

# set if data shold be read in advance
fileread = True
ensemble_mode = False 

print('fileread = ', fileread)
print('ensemble_mode = ', ensemble_mode)

nb_classes = 82
sess = tf.Session()

# read and save to easy to load file if
# fileread is set to true
if fileread:
	x_data = np.load('x_data.npy')
	y_data = np.load('y_data.npy')
	labels_names = np.load('labels.npy')

models = []
num_models = 7

if ensemble_mode:
	for m in range(num_models):
		models.append(Model(sess, 'model' + str(m), nb_classes))
else:
	model1 = Model(sess, 'model1', nb_classes)

training_epochs = 40
batch_size = 100
num_train_files = load_data.get_num_train_files()
print("num_train_files: ", num_train_files)
# to avoid compile error when filread is false	
end = 0
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('./logs/cnn_math_logs_false_100ep_20bs_r1')
writer.add_graph(sess.graph)

for epoch in range(training_epochs):
	batch_index = 0
	avg_cost = 0
	avg_cost_list = np.zeros(num_models)
	total_batch = int(num_train_files / batch_size)
	#print("Total batch: ", total_batch)

	for i in range(total_batch):
		# if data is already read from file
		if fileread:
			end = batch_index + batch_size
			# to not get out of range
			if(end > num_train_files):
				end = num_train_files - 1
			batch_xs = x_data[batch_index:end] 
			batch_ys= y_data[batch_index:end]
		else:
			# otherwise read data
			batch_xs, batch_ys = load_data.get_jpeg_data(sess, batch_size, nb_classes)
		if(ensemble_mode):
			for m_idx, m in enumerate(models):
				s, c, _ = m.train(batch_xs, batch_ys)	
				avg_cost_list[m_idx] += c / total_batch
				writer.add_summary(s, global_step=i)
		else:
			s, c, _ = model1.train(batch_xs, batch_ys)
			avg_cost += c / total_batch
			writer.add_summary(s, global_step=i)
		batch_index = end
		print('processing {:.1f}%'.format((i / total_batch) * 100))
	if ensemble_mode:
		print("Epoch: ", "%04d" % (epoch + 1), "cost = ", avg_cost_list)
	else:
		print("Epoch: ", "%04d" % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost))

pictures, labels = load_data.get_all_data(sess, nb_classes, savefile=False, trainset=False)
num_test_files = load_data.get_num_test_files()

r = random.randint(0,  num_test_files - 1)

if ensemble_mode:
	predictions = np.zeros(num_test_files * nb_classes).reshape(num_test_files, nb_classes)

	for m_idx, m in enumerate(models):
		print(m_idx, 'Accuracy: ', m.get_accuracy(pictures, labels))
		p = m.predict(pictures)
		predictions += p

	ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
	ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
	print('Ensemble accuracy: ', sess.run(ensemble_accuracy))
else:
	print("Label: ", load_data.labels[model1.get_label(labels[r:r+1])])
	print("Prediction: ", load_data.labels[model1.predict(pictures[r:r+1])])

	# Uncomment one of the first two and the third line to view the test image
	#plt.imshow(pictures[r:r + 1].reshape(45, 45, 1), cmap="Greys", interpolation="nearest") 	
	#plt.imshow(np.reshape(pictures[r:r+1], (45, 45, 1)))
	#plt.show()

	print("Accuracy: ", model1.get_accuracy(pictures, labels))
	print("Precision: ", model1.get_precision(pictures, labels))
