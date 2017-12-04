import tensorflow as tf

"""
solve a typical line equation  y = Ax + b for 2 sets of values of x and b.
"""
if __name__ == "__main__":
    A = tf.constant([100,1000], name="A")
    x = tf.placeholder(shape=[2], dtype=tf.int32, name="x")
    b = tf.placeholder(shape=[2],dtype=tf.int32, name="y")

    Ax = tf.multiply(A, x, name="Ay")
    y = tf.add(Ax, b, name="y")

    # compute 2 different values of y using 2 different values of x and b, A is constant
    with tf.Session() as sess:
        print(sess.run(fetches=y,feed_dict={x:[2,5], b:[6,7]}))
        #print("y = ", y)

