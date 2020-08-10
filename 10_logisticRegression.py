import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
from tensorflow import name_scope
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Tensorflow逻辑回归模型
with tf.name_scope("wx_b"):
    y_hat = tf.nn.softmax(tf.matmul(x, W) + b)

W_history = tf.summary.histogram("Weights:", W)
b_history = tf.summary.histogram("biases:", b)

with tf.name_scope("cross-entropy") as scope:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
    tf.summary.scalar('cross-entropy', loss)

with tf.name_scope('Train') as scope:
    optimazer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

y_pred = tf.arg_max(tf.nn.softmax(tf.matmul(x, W) + b), dimension=1)
y_result = tf.arg_max(y, dimension=1)


init = tf.global_variables_initializer()

merged_summary_opt = tf.summary.merge_all()

max_epoches = 50
batch_size = 100
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter('graphs', sess.graph)
    for epoch in range(max_epoches):
        loss_avg = 0
        num_of_batch = int(mnist.train.num_examples/batch_size)
        print(num_of_batch)
        for i in range(num_of_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, l, summary_str = sess.run([optimazer, loss, merged_summary_opt], feed_dict={x: batch_xs, y: batch_ys})
            loss_avg += 1
            summary_writer.add_summary(summary_str, epoch*num_of_batch + i)
            loss_avg = loss_avg/num_of_batch
        print('Epoch {0}: Loss {1}'.format(epoch, l))
    print('Done.')
    pred, result = sess.run([y_pred, y_result], feed_dict={x: mnist.test.images, y: mnist.test.labels})
    accure = 0
    for i in range(len(pred)):
        if pred[i] == result[i]:
            accure += 1
    print('pred: {0}, result: {1}, accuracy: {2}'.format(pred, result, accure/len(pred)))

