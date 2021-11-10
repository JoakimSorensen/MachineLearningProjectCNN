"""
Created by Joakim Sorensen
2017008298
Machine Learning 2017
Kyung Hee University

"""

import load_data
import tensorflow as tf

"""
This script is used to create the npy
files used by cnn_math_main.py when set to
read from file.
"""
sess = tf.Session()
load_data.get_all_data(sess, 82, savefile=True, trainset=True)
