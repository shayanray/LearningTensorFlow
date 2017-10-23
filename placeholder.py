import tensorflow as tf

x = tf.placeholder(dtype=tf.int32 ,shape=[3], name="x")
y = tf.placeholder(dtype=tf.int32, shape=[3], name="y")

sum_x = tf.reduce_sum(x, name="sum_x")
prod_y = tf.reduce_prod(y, name="prod_y")

session = tf.Session()

print(session.run(sum_x, feed_dict={x:[10,20,30]}))
print(session.run(prod_y, feed_dict={y:[40,36,23]}))

writer = tf.summary.FileWriter("./placeholders", session.graph)

writer.close()
session.close()
