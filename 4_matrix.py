import tensorflow as tf

# 交互式会话
sess = tf.InteractiveSession()
I_matrix = tf.eye(4)

# 使用eval()方法取出张量值
print(I_matrix.eval())

# 变量
X_matrix = tf.Variable(tf.constant([[2, 4, 2], [2, 6, 3]]))
X_matrix.initializer.run()
print(X_matrix.eval())