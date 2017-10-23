import tensorflow as tf
import numpy as np

session = tf.Session()
a = np.array(50, dtype=np.int32)
b = np.array([1,2,3,4], dtype=np.int32)

print(" rank of a (scalar)  >> ", session.run(tf.rank(a)))
print("shape of a >> ", session.run(tf.shape(a)))
print("rank of b >> ", session.run(tf.rank(b)))
print("shape of b >> ", session.run(tf.shape(b)))


