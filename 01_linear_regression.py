import tensorflow as tf
import numpy as np

# Script shows how to use TensorFlow for linear regression for one feature and one class
# and use batches for the training

# CUSTOMIZABLE: Collect/Prepare data
datapoint_size = 1000
batch_size = 10
steps = 1000
actual_W = 2
actual_b = 10
learning_rate = 0.01

# Fake data
all_xs = []
all_ys = []
for i in range(datapoint_size):
    # Create fake data for y = W.x + b where W = actual_W, b = actual_b
    all_xs.append(i % 10)
    all_ys.append(actual_W*(i % 10)+actual_b)
all_xs = np.transpose([all_xs])
all_ys = np.transpose([all_ys])

# Model construction
# placeholder for input variable
x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')

# definition of weight matrix and bias vector
W = tf.Variable(initial_value=tf.zeros([1, 1]), name='weight')
b = tf.Variable(initial_value=tf.zeros([1]), name='bias')

# definition of outcome variable ( y = W * x + b )
y_pred = tf.add(x=tf.matmul(a=x, b=W), y=b)

# placeholder for true outcome values
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_true')

# definition for cost function (loss function ...) sum((yt -yp)^2)
cost = tf.reduce_mean(input_tensor=tf.square(x=(y_true - y_pred)), name='cost_func')

# definition for gradient decent object with step == learning_rate and cost as minimization target
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=cost)

# for performance reasons and other things Tensorflow has to be executed within session
sess = tf.Session()

# define variable initialisation
init = tf.global_variables_initializer()

# perform actual variable initialisation
sess.run(fetches=init)

# training steps
for i in range(steps):
    # create batch of input and true data
    if datapoint_size == batch_size:
        batch_start_idx = 0
    elif datapoint_size < batch_size:
        raise ValueError("Datapoint size: {datapoint_size}, must be greater than batch size: {batch_size}".format(
            datapoint_size=datapoint_size,
            batch_size=batch_size))
    else:
        batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)

    batch_end_idx = batch_start_idx + batch_size
    batch_xs = all_xs[batch_start_idx:batch_end_idx]
    batch_ys = all_ys[batch_start_idx:batch_end_idx]
    xs = np.array(batch_xs)
    ys = np.array(batch_ys)

    # training
    feed = {x: xs, y_true: ys}
    sess.run(fetches=train_step, feed_dict=feed)

    print("Iteration {iter} and cost is {cost} ".format(iter=i, cost=sess.run(fetches=cost, feed_dict=feed)))

print("Predicted: W={W} and  b={b}".format(W=sess.run(fetches=W), b=sess.run(fetches=b)))
