import tensorflow as tf

# create constants for as tensors
num1 = tf.constant(7, name="constant_7", dtype=tf.int32)
num2 = tf.constant(8, name="constant_8", dtype=tf.int32)
num3 = tf.constant(69, name="constant_69", dtype=tf.int32)
num4 = tf.constant(23, name="constant_23", dtype=tf.int32)

# create nodes of computation
mul = tf.multiply(num1, num2, name="multiply")
div = tf.div(num3, num4, name="divide")

# create final node
result = tf.add_n([mul, div], name="add_n")

#create session to supervise the execution of the graph
session = tf.Session()
# run the graph in the session along with results
print(session.run(result))

# write the graph to a log file
writer = tf.summary.FileWriter('./graph_output', session.graph)
writer.close()

# tensorboard --logdir="./graph_output" # use it to visualize the graph, it reads from the log file.

# session has to be closed
session.close()
