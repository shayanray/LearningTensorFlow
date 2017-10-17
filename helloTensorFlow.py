import tensorflow as tf

# create a constant
hello = tf.constant("Hello TensorFlow !!")

# everything works out of a session in tf
session = tf.Session()

#print
print(session.run(hello))


