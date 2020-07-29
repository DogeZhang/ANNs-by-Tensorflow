"""
张量(Tensor)：可理解为一个 n 维矩阵，所有类型的数据，包括标量、矢量和矩阵等都是特殊类型的张量。
    1. 常量：不解释
    2. 变量：使用时需要*显式*初始化，占用内存，可以分开存储，可以存储在参数服务器上
    3. 占位符：用于将值输入 TensorFlow 图中。
        它们可以和 feed_dict 一起使用来输入数据。
        在训练神经网络时，它们通常用于提供新的训练样本。
        在会话中运行计算图时，可以为占位符赋值。这样在构建一个计算图时不需要真正地输入数据。
        需要注意的是，占位符不包含任何数据，因此不需要初始化它们。
"""
import tensorflow as tf

"""----------------------------------常量----------------------------------"""
# 声明一个常量：
name = tf.constant("zhang")

# 声明一个标量常量：
a = tf.constant(6)
b = tf.constant([1, 5, 3])

"""
依照维度声明元素为0的矩阵：
    形如[m, n]的矩阵：
    tf.zeros([m, n], tf.dtype)
    dtype 可以是int32、float32等
    与之类似的有
    tf.ones
"""
zeros = tf.zeros([2,3],tf.int32)
# [[0,0,0],[0,0,0]]

# like
c_like_a = tf.zeros_like(a) # [0]
d_like_b = tf.ones_like(b) # [0, 0, 0]

"""
linspace 生成一个等差序列
    tf.linspace(start, stop, num)
        start: 起始数字 
        stop: 结束数字
        num: 数列大小
    可知等差序列中的差为： (stop - start)/(num - 1)
"""

"""
random 
——正态分布 random_normal——
使用以下语句创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的——正态分布——随机数组：
    tf.random.normal([m, n, ...], mean=, stddev=, seed=)
        [m, n....] 矩阵形状
        mean 均值（默认0）
        stddev 标准差（默认1.0）

——截尾正态分布 random.truncated_normal——        
创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的——截尾正态分布——随机数组：
    tf.truncated_normal([m,n], stddev=, seed=)

——伽马分布 random.uniform——
要在种子的 [minval（default=0），maxval] 范围内创建形状为 [M，N] 的给定——伽马分布——随机数组，请执行如下语句：
    tf.random_uniform([m,n], maxval=, seed=)
"""
normal = tf.random.normal([5, 7], mean=1, stddev=2, seed=12)
truncated_normal = tf.random.truncated_normal([2, 3], stddev=1, seed=12)
random_uniform = tf.random.uniform([2, 3], maxval=4, seed=12)

"""
裁剪 crop、shuffle 常常可用在图像剪裁中
image.random_crop: 从目标矩阵中随机截取目的大小的矩阵
random.shuffle：沿着它的第一维随机排列张量
"""
croped_nromal = tf.image.random_crop(normal,[2,5],seed=12)
shuffled_truncated_normal = tf.random.shuffle(truncated_normal)

"""
随机生成的张量受初始种子值的影响。要在多次运行或会话中获得相同的随机数，应该将种子设置为一个常数值。
当使用大量的随机张量时，可以使用 tf.set_random_seed() 来为所有随机产生的张量设置种子。
以下命令将所有会话的随机张量的种子设置为 54：
tf.set_random_seed(54)
"""

"""----------------------------------变量----------------------------------"""
variable_a = tf.Variable(random_uniform)
variable_b = tf.Variable(random_uniform)

# 指定一个变量来初始化另一个变量
variable_c = tf.Variable(variable_a.initialized_value(), name="vc")


"""----------------------------------placeholder----------------------------------"""
""" 
tf.placeholder(dtype,shape=None,name=None)
dtype 定占位符的数据类型，并且必须在声明占位符时指定。
"""
# 为 x 定义一个占位符并计算 y=2*x，使用 feed_dict 输入一个随机的 4×5 矩阵：
x = tf.placeholder("float")
y = 2 * x
data = tf.random.uniform([4, 5], 10)
with tf.Session() as sess:
    x_data = sess.run(data)
    print(sess.run(y, feed_dict={x:x_data}))

"""
Tips:
所有常量、变量和占位符将在代码的计算图部分中定义。如果在定义部分使用 print 语句，只会得到有关张量类型的信息，而不是它的值。
为了得到相关的值，需要创建会话图并对需要提取的张量显式使用运行命令，如:
print(sess.run(a))
Will print the value of a defined in step 1
"""
