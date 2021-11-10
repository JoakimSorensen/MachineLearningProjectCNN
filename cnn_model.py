"""
Created by Joakim Sorensen
2017008298
Machine Learning 2017
Kyung Hee University

"""

import tensorflow as tf
from sklearn import metrics

"""
The model for the CNN, built with tensorflow.
Can be trained and tested with functions defined
below.
"""
class Model:


	def __init__(self, sess, name, nb_classes):
		self.name = name
		self.sess = sess
		self.nb_classes = nb_classes
		self._build_net()

	"""
	Builds the CNN according to the img_size.
	New image size will need new values for the layers,
	changing just the image size will return in exceptions.
	"""
	def _build_net(self):
		with tf.variable_scope(self.name):
			img_size = 45 * 45 * 1

			self.X = tf.placeholder(tf.float32, [None, img_size])
			self.Y = tf.placeholder(tf.float32, [None, self.nb_classes])

			# define layers
			input_layer = tf.reshape(self.X, [-1, 45, 45, 1])
			with tf.variable_scope('conv_pool_1'):

				conv1 = tf.layers.conv2d(inputs=input_layer, 
																filters=81, 
																kernel_size=[5, 5], 
																padding='same', 
																activation=tf.nn.relu)

				pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=3)

				tf.summary.image('input1', input_layer, 3)
				tf.summary.histogram('conv1', conv1)
				tf.summary.histogram('pool1', pool1)
			
			with tf.variable_scope('conv_pool_2'):
				conv2 = tf.layers.conv2d(inputs=pool1, 
																filters=162, 
																kernel_size=[5, 5], 
																padding='same', 
																activation=tf.nn.relu)

				pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=3)
				
				#tf.summary.image('input2', pool1, 3)
				tf.summary.histogram('conv2', conv2)
				tf.summary.histogram('pool2', pool2)

			with tf.variable_scope('conv_pool_3'):
				conv3 = tf.layers.conv2d(inputs=pool2, 
																filters=243, 
																kernel_size=[5, 5], 
																padding='same', 
																activation=tf.nn.relu)

				pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[5, 5], strides=5)

				#tf.summary.image('input3', pool2, 3)
				tf.summary.histogram('conv3', conv3)
				tf.summary.histogram('pool3', pool3)

			with tf.variable_scope('dense'):
				#pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 162])
				pool2_flat = tf.reshape(pool3, [-1, 1 * 1 * 243])

				dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
				dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

				logits = tf.layers.dense(inputs=dropout, units=self.nb_classes)
				
				#tf.summary.image('input4', pool3, 3)
				tf.summary.histogram('pool2_flat', pool2_flat)
				tf.summary.histogram('dense', dense)
				tf.summary.histogram('dropout', dropout)
				tf.summary.histogram('logits', logits)

			with tf.variable_scope('predictions'):
			# predictions
				classes = tf.argmax(input=logits, axis=1)
				self.probabilities = tf.nn.softmax(logits, name='softmax_tensor')
				tf.summary.histogram('classes', classes)
				tf.summary.histogram('probabilities', self.probabilities)

			self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.Y, logits=logits)
			self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

			is_correct = tf.equal(tf.argmax(self.probabilities, 1), tf.argmax(self.Y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
			#confusion_matrix(tf.argmax(self.probabilities, 1), tf.argmax(self.Y, 1))	
			predicted = tf.argmax(self.probabilities, 1)
			true = tf.argmax(self.Y, 1)
			tp = tf.count_nonzero(predicted * true, dtype=tf.float32)
			fp = tf.count_nonzero(predicted * (true - 1), dtype=tf.float32)
			self.precision = tf.divide(tp, tp + fp)
			
			cost_summ = tf.summary.scalar('loss', self.loss)
			acc_summ = tf.summary.scalar('accuracy', self.accuracy)
			prec_summ = tf.summary.scalar('precision', self.precision)

			self.summary = tf.summary.merge_all()

	"""
	Train the model with given datasets.
	x_data: a tensorflow vector with image data
	y_data: a one hot encoded tensorflow vector for labels
	returns loss and optimizer output
	"""
	def train(self, x_data, y_data):
		return self.sess.run([self.summary, self.loss, self.optimizer], 
						feed_dict = {self.X: x_data, self.Y: y_data})	

	"""
	Predict the labels for given data.
	x_test: a tensorflow vector with image data
	returns index of predicted label
	"""
	def predict(self, x_test):
		return self.sess.run(tf.argmax(self.probabilities, 1), 
																					feed_dict={self.X: x_test})[0]
	"""
	Returns the labels as a string.
	y_data: one hot encoded tensorflow vector
	returns corresponding string label
	"""
	def get_label(self, y_data):
		return self.sess.run(tf.argmax(y_data, 1))[0]	

	"""
	Returns the accuracy of the model with
	given test data.
	x_test: a tensorflow vector with image data
	y_test: the correct labels as one hot encoded
					tensorflow vector
	returns the accuracy 0 <= accuracy <= 1
	"""
	def get_accuracy(self, x_test, y_test):
		return self.accuracy.eval(session=self.sess, feed_dict={self.X: x_test, self.Y: y_test})
	

	def get_precision(self, x_test, y_test):
		predicted = self.sess.run(tf.argmax(self.probabilities, 1), feed_dict={self.X: x_test})
		true = self.sess.run(tf.argmax(y_test, 1))
		tp = tf.count_nonzero(predicted * true)
		fp = tf.count_nonzero(predicted * (true - 1))
		#precision = tp / (tp + fp)
		precision = tf.divide(tp, tp + fp)
		return self.sess.run(precision, feed_dict={self.X: x_test, self.Y:y_test}) 
