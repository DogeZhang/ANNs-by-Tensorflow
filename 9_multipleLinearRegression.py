import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab


def normalize(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    matrix = (matrix - mean)/std
    return matrix


boston = tf.contrib.learn.datasets.load_dataset('boston')
X_train = boston.data
X_train = normalize(X_train)
Y_train = boston.target

m = len(X_train)
n = 13
X = tf.placeholder(tf.float32, shape=[m, n])
Y = tf.placeholder(tf.float32)

w = tf.Variable(tf.random.normal([n, 1]))
b = tf.Variable(tf.zeros([m, 1]))
Y_hat = tf.matmul(X, w) + b

loss = tf.reduce_mean(tf.square(Y - Y_hat))
lossRecord = []

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('graph', sess.graph)
    for i in range(200):
        _, lss = sess.run([optimizer, loss], feed_dict={X: X_train, Y: Y_train})
        lossRecord.append(lss)
        print('Epoch: {0} ｜ Loss: {1}'.format(i, lss))
    writer.close()
    # print('weight: {0} | bais: {1}'.format(weight, bais))

    plt.plot(lossRecord)
    plt.show()
