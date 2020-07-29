"""
tensorflow将程序分为两个独立的部分：
第一部分：
    神经网络蓝图
    即：Tensor
第二部分：
    神经网络的运算
    即：Operation Object
"""
import tensorflow as tf

"""
计算图：是包含节点和边的网络。本节定义所有要使用的数据，也就是张量（tensor）对象（常量、变量和占位符）
第一部分：定义一个简单的结构:
    a 
      \
        + -> [_add]
      /
    b   
"""
a = tf.constant([2,4,2,5])
b = tf.constant([4,6,2,4])
_add = tf.add(a, b)

"""
第二部分：运行
定义要执行的所有计算，即运算操作对象（Operation Object，简称 OP）。
"""

with tf.Session() as sess:
    print(sess.run(_add))

"""
相同的结果：
sess = tf.Session()
print(sess.run(_add))
sess.close()
"""

"""
输出结果：
[ 6 10  4  9]
"""