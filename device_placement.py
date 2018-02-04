import tensorflow as tf

from tensorflow.python.client import device_lib

gpus = [d for d in device_lib.list_local_devices()
        if d.device_type == 'GPU']
print([g.name for g in gpus])

# Creates a graph.
with tf.device(gpus[1].name):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
