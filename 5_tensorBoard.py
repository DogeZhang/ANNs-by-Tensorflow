"""
TensorBoard
使用 TensorBoard 来提供计算图形的图形图像。这使得理解、调试和优化复杂的神经网络程序变得很方便。
TensorBoard 也可以提供有关网络执行的量化指标。
它读取 TensorFlow 事件文件，其中包含运行 TensorFlow 会话期间生成的摘要数据。

几个典型用法：
    记录网络学习过程中损失函数的变化
    可视化梯度、权重或特定层的输出分布
"""

import tensorflow as tf
"""
记录损失函数随时间变化：
    loss = tf...
    tf.summary.scalar('loss', loss)

可视化特定层的输出分布：
    output_tensor = tf.matmul(input_tensor, weights) + biases
    tf.summary.histogram('output', output_tensor)
    
生成的所有摘要将写在 summary_dir 目录中
    tf.summary.Filewriter:
        writer = tf.summary.Filewriter('summary_dir', sess.graph)
        
        
最后在命令行：
    tensorboard --logdir = summary_dir
"""
