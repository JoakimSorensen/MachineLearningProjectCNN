import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt
import load_data
import numpy as np

# 82 expressions
nb_classes = 82
# Math expr jpeg image of shape 45 * 45 = 2025
img_size = 45 * 45 * 3

x_data = np.load('x_data.npy')
y_data = np.load('y_data.npy')
labels_names = np.load('labels.npy')

X = tf.placeholder(tf.float32, [None, img_size])
Y = tf.placeholder(tf.float32, [None, nb_classes])

input_layer = tf.reshape(X, [-1, 45, 45, 3])

conv1 = tf.layers.conv2d(inputs=input_layer, 
												filters=81, 
												kernel_size=[5, 5], 
												padding='same', 
												activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=3)

conv2 = tf.layers.conv2d(inputs=pool1, 
												filters=162, 
												kernel_size=[5, 5], 
												padding='same', 
												activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3)

pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 162])

dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

logits = tf.layers.dense(inputs=dropout, units=nb_classes)

# predictions
classes = tf.argmax(input=logits, axis=1)
probabilities = tf.nn.softmax(logits, name='softmax_tensor')

loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

is_correct = tf.equal(tf.argmax(probabilities, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 10
batch_size = 20
num_train_files = load_data.get_num_train_files()
batch_index = 0
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(num_train_files / batch_size)
		#print("Total batch: ", total_batch)

		for i in range(total_batch):
			#end = batch_index + batch_size
			#if(end > num_train_files - 1):
			#	end = num_train_files - 1
			#batch_xs = x_data[batch_index:end] 
			#batch_ys= y_data[batch_index:end]
			batch_xs, batch_ys = load_data.get_jpeg_data(sess, batch_size, nb_classes)
			c, _ = sess.run([loss, optimizer], feed_dict = {X: batch_xs, Y: batch_ys})
			avg_cost += c / total_batch
			batch_index = end
			print('processing {:.1f}%'.format((i / total_batch) * 100))
		print("Epoch: ", "%04d" % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost))
	
	pictures = x_data
	labels = y_data
	#load_data.labels = labels_names
	#pictures, labels = load_data.get_test_data(sess, nb_classes)
	r = random.randint(0, num_train_files - 1)
	print("Label: ", load_data.labels[sess.run(tf.argmax(labels[r:r+1], 1))[0]])
	print("Prediction: ", load_data.labels[sess.run(tf.argmax(probabilities, 1), 
												feed_dict={X: pictures[r:r+1]})[0]])

	plt.imshow(pictures[r:r + 1].reshape(45, 45, 3), cmap="Greys", interpolation="nearest") 	
	#plt.imshow(np.reshape(pictures[r:r+1], (45, 45, 3)))
	plt.show()

	print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: pictures, Y: labels}))
