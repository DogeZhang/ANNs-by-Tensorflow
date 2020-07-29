import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def nomarlize(X):
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X


boston = tf.contrib.learn.datasets.load_dataset('boston')
X_train = boston.data[:, 5]
Y_train = boston.target

n_samples = len(X_train)

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

b = tf.Variable(0.0)
w = tf.Variable(0.0)

Y_hat = X * w + b

# loss = tf.square(Y_hat - Y_train) Wrong!
loss = tf.square(Y - Y_hat)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

init_op = tf.global_variables_initializer()
total = []

with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter('graph', sess.graph)

    for i in range(100):
        total_loss = 0
        for x, y in zip(X_train, Y_train):
            lss = sess.run([optimizer, loss, Y_hat, Y], feed_dict={X:x, Y:y})
            total_loss += 1
        total.append(total_loss / n_samples)
        # print('Epoch {0}: Loss {1}'.format(i, total_loss / n_samples))
        print('Epoch {0}: Loss: {1}'.format(i, lss))
    writer.close()
    b_value = sess.run(w)
    w_value = sess.run(w)
    print("b_value: {0}, w_value: {1}".format(b_value, w_value))


