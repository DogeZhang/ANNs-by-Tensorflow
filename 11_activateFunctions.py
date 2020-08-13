"""
6种激活函数：
1- 阈值激活函数
2- Sigmoid激活函数
3- 双曲正切激活函数
4- 线性激活函数
5- 整流线性单元（ReLU）激活函数
6- Softmax激活函数
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
1-阈值激活函数
    最简单，如果神经元的激活值大于某一值，则激活，否则静默。
"""


def threshold(x):
    cond = tf.less(x, tf.zeros(tf.shape(x), dtype=x.dtype))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out


h = np.linspace(-1, 1, 50)
out = threshold(h)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    y = sess.run(out)
    plt.xlabel('Activity of Neuron')
    plt.ylabel('Output of Neuron')
    plt.title('Threshold Activation Function')
    plt.plot(h, y)
    plt.show()
