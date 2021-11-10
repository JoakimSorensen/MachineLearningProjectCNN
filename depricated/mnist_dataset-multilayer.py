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

X = tf.placeholder(tf.float32, [None, img_size])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# Might be excessive, doesn't affect result much
W0 = tf.Variable(tf.truncated_normal([img_size, 100], stddev=0.1), name="weight0")
#W0 = tf.Variable(tf.random_normal([2025, 100]), name="weight0")
b0 = tf.Variable(tf.random_normal([100]), name="bias0")
layer0 = tf.nn.relu(tf.matmul(X, W0) + b0)

# zero mean Gaussian distribution with standard deviation
stddev1 = np.sqrt(2 / np.prod(W0.get_shape().as_list()[1:]))

W1 = tf.Variable(tf.truncated_normal([100, 100], stddev=stddev1), name="weight1")
#W1 = tf.Variable(tf.random_normal([100, 100]), name="weight1")
b1 = tf.Variable(tf.random_normal([100]), name="bias1")
layer1 = tf.nn.relu(tf.matmul(layer0, W1) + b1)

stddev2 = np.sqrt(2 / np.prod(W1.get_shape().as_list()[1:]))

W2 = tf.Variable(tf.truncated_normal([100, 100], stddev=stddev2), name="weight2")
#W2 = tf.Variable(tf.random_normal([100, 100]), name="weight2")
b2 = tf.Variable(tf.random_normal([100]), name="bias2")
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

stddev3 = np.sqrt(2 / np.prod(W2.get_shape().as_list()[1:]))

W3 = tf.Variable(tf.truncated_normal([100, nb_classes], stddev=stddev3),name="weight3")
#W3 = tf.Variable(tf.random_normal([100, nb_classes]), name="weight3")
b3 = tf.Variable(tf.random_normal([nb_classes]), name="bias3")
hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.37).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 1
batch_size = 20
num_train_files = load_data.get_num_train_files()	
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(num_train_files / batch_size)
		#print("Total batch: ", total_batch)

		for i in range(total_batch):
			batch_xs, batch_ys = load_data.get_jpeg_data(sess, batch_size, nb_classes)
			c, _ = sess.run([cost, optimizer], feed_dict = {X: batch_xs, Y: batch_ys})
			avg_cost += c / total_batch
			print('processing {:.1f}%'.format((i / total_batch) * 100))
		print("Epoch: ", "%04d" % (epoch + 1), "cost = ", "{:.9f}".format(avg_cost))

	pictures, labels = load_data.get_test_data(sess, nb_classes)
	r = random.randint(0, num_train_files - 1)
	print("Label: ", load_data.labels[sess.run(tf.argmax(labels[r:r+1], 1))[0]])
	print("Prediction: ", load_data.labels[sess.run(tf.argmax(probabilities, 1), 
												feed_dict={X: pictures[r:r+1]})[0]])

	plt.imshow(pictures[r:r + 1].reshape(28, 28), cmap="Greys", interpolation="nearest") 	
	#plt.imshow(np.reshape(pictures[r:r+1], (45, 45, 3)))
	plt.show()

	#r = random.randint(0, mnist.test.num_examples - 1)
	#print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
	#print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: load_data.get_jpeg_data(sess, load_data.get_num_test_files(), num_classes, train=False)[r:r + 1]}))

	#plt.imshow(load_data.get_jpeg_data(sess, load_data.get_num_test_files(), num_classes, train=False)[r:r + 1].reshape(45, 45), cmap="Greys", interpolation="nearest")
	#plt.show()

	#print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
