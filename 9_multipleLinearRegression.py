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
# 将w, b转换成稀疏矩阵，即非全连接
w_sparse = tf.SparseTensor(indices=[[1, 0], [2, 0], [6, 0], [10, 0]], values=tf.identity(w), dense_shape=[n, 1])
# b_sparse = tf.SparseTensorValue(indices=)
# w_dense = tf.sparse_to_dense([[1, 0], [2, 0], [6, 0], [10, 0]], output_shape=[n, 1], sparse_values=0.0)
Y_hat = tf.matmul(X, w_sparse) + b

loss = tf.reduce_mean(tf.square(Y - Y_hat))
lossRecord = []

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('graph', sess.graph)
    for i in range(200):
        _, lss, w = sess.run([optimizer, loss, w_sparse], feed_dict={X: X_train, Y: Y_train})
        lossRecord.append(lss)
        print('Epoch: {0} ｜ Loss: {1}'.format(i, lss))
        print('W: {0}'.format(w))
    writer.close()
    # print('weight: {0} | bais: {1}'.format(weight, bais))

    plt.plot(lossRecord)
    plt.show()
