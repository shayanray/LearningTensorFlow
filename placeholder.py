import tensorflow as tf

x = tf.placeholder(dtype=tf.int32 ,shape=[3], name="x") # 1d vector with 3 elements
y = tf.placeholder(dtype=tf.int32, shape=[3], name="y") # if shape is not specified, it can hold any shape

sum_x = tf.reduce_sum(x, name="sum_x")
prod_y = tf.reduce_prod(y, name="prod_y")
final_mean = tf.reduce_mean([sum_x, prod_y], name="final_mean")

session = tf.Session()

print(session.run(sum_x, feed_dict={x:[10,20,30]}))
print(session.run(prod_y, feed_dict={y:[40,36,23]}))
print(session.run(final_mean, feed_dict={x:[10,20,30], y:[40,36,23]}))

writer = tf.summary.FileWriter("./placeholders", session.graph)

writer.close()
session.close()
