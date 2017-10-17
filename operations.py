import tensorflow as tf

num1 = tf.constant(7, name="constant_7", dtype=tf.int32)
num2 = tf.constant(8, name="constant_8", dtype=tf.int32)
num3 = tf.constant(69, name="constant_69", dtype=tf.int32)
num4 = tf.constant(23, name="constant_23", dtype=tf.int32)

mul = tf.multiply(num1, num2, name="multiply")
div = tf.div(num3, num4, name="divide")

result = tf.add_n([mul, div], name="add_n")

session = tf.Session()
print(session.run(result))


session.close()
