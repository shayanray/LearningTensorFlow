import tensorflow as tf

"""

y = Wx + b

"""


if __name__ == "__main__":
    W = tf.Variable([0.3], dtype=tf.float32, name="W")
    b = tf.Variable([0.5], dtype=tf.float32, name="b")

    x = tf.placeholder(dtype=tf.float32, name="x")

    Wx = tf.multiply(W, x, name="Wx")
    model = tf.add(Wx, b, name="model")

    y = tf.placeholder(dtype=tf.float32, name="y")

    # loss function
    lossfn = tf.reduce_sum(tf.square(model - y))

    #optimizer gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.01) # learning rate

    # find minimum residual sum of squares - loss function
    train = optimizer.minimize(lossfn)

    init = tf.global_variables_initializer()

    x_train = [0.1,0.2,0.3]
    y_train = [0.5,0.9,1.7]
    with tf.Session() as sess:
        sess.run(init)

        for i in range(10):
            sess.run(train, feed_dict={x:x_train, y:y_train})

        currW, currB, currLoss = sess.run([W,b,lossfn], feed_dict={x:x_train, y:y_train})
        print("currW >> ", currW, "currB >> ", currB, "currLoss >> ", currLoss, )