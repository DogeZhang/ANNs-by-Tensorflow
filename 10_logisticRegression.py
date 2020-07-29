import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 734])
y = tf.placeholder(tf.float32, )