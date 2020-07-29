# 引入tensorflow
import tensorflow as tf
# 创建一个常量，内容为hello zhang
message = tf.constant('Hello Zhang!')

# 创建一个会话来执行代码，类似C语言的main()函数
with tf.Session() as sess:
    print(sess.run(message).decode())

# 这里的with打开可以避免忘记使用close方法关闭会话的问题。