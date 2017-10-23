import tensorflow as tf

a = tf.constant([100, 200, 300], name="oned_array")
b = tf.constant([1,2,3], name="another_oned_array")

rsum = tf.reduce_sum(a, name="reduced_sum")
rproduct = tf.reduce_prod(b, name="reduced_product")

rmean = tf.reduce_mean([rsum, rproduct], name="reduced_mean")

session = tf.Session()
print(session.run(rsum))
print(session.run(rproduct))
print(session.run(rmean))

writer = tf.summary.FileWriter("./graph_output3", session.graph)

writer.close()
session.close()

