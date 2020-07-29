"""
在 TensorFlow 中可以通过三种方式读取数据：
    1. 通过feed_dict传递数据；
    2. 从文件中读取数据；
    3. 使用预加载的数据；
"""
import tensorflow as tf

"""
使用feed_dict传递数据 （在内存中传递）
    运行每个步骤时都会使用 run() 或 eval() 函数调用中的 feed_dict 参数来提供数据。
    这是在占位符的帮助下完成的，这个方法允许传递 Numpy 数组数据。
"""

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

with tf.Session() as sess:
    X_Array = [2,3,4] # some Numpy Array
    Y_Array = [2,3,4] # some Numpy Array
    loss = tf.constant(42)
    sess.run(loss, feed_dict = {x: X_Array, y: Y_Array})

"""
从文件中读取 （适用于大型数据文件）
    1 使用字符串张量 ["file0", "file1"] 
"""
files = tf.train.match_filenames_once('*.jpg')
"""
    2 文件队列
        创建一个队列名来保存文件名，使用tf.train.string_input_producer
"""
filename_queue = tf.train.string_input_producer(files)
"""
    3 Reader 用于从文件名队列中读取文件。
        read方法是标识文件和记录（调试时有用）以及标量字符串值的关键字。
"""
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
"""
    4 Decoder
        使用一个或多个解码器和转换操作将字符串解码为构成训练样本的张量。
"""
record_defaults = [[1], [1], [1]]
col1, col2, col3 = tf.decode_csv(value, record_defaults = record_defaults)