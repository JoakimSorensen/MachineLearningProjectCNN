import tensorflow as tf
import load_data as ld
import matplotlib.pyplot as plt
import random
import numpy as np

sess = tf.Session()
#pictures, labels = ld.get_jpeg_data(sess, 100, 82)
pictures, labels = ld.get_test_data(sess, 82, savefile=True)

num_train_files = ld.get_num_train_files()	
r = random.randint(0, num_train_files - 1)
#plt.imshow(pictures[r:r + 1].reshape(45, 45, 3), cmap="Greys", interpolation="nearest") 	
plt.imshow(np.reshape(pictures[r:r+1], (45, 45, 3)))
plt.show()
#ld.get_test_data(sess, 82)

