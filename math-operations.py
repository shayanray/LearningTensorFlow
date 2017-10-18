import tensorflow as tf

a = tf.constant(5.0, name="constant_a")
b = tf.constant(6.0, name="constant_b")
c = tf.constant(64.0, name="constant_c")
m = tf.constant([[1,2],[3,4]], name= "matrix")

square_val = tf.square(a,name="square_op")
sqrt_val = tf.sqrt(c, name="sqrt_op")
pow_val = tf.pow(a,b,name="power_op")

add_all_val = tf.add_n([square_val, sqrt_val, pow_val], name="add_all")
session = tf.Session()

print("square of a ", session.run(a) ," >> ",session.run(square_val))
print("square root of b ", session.run(b) ," >> ",session.run(sqrt_val))
print("a to the power of b  >> ", session.run(pow_val))
print("add all the above values >> ",session.run(add_all_val))

print("rank of add all value is >> " , session.run(tf.rank(add_all_val))) # scalar has dimension 0
print("rank of matrix m is >> ", session.run(tf.rank(m))) # number of dimensions

writer = tf.summary.FileWriter('./graph_output2', session.graph)

writer.close()
session.close()


