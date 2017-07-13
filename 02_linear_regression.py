import tensorflow as tf
import numpy as np

# Script shows how to use TensorFlow for linear regression for multiple features and one class
# and use batches for the training. In addition, create and save computational graphs and stuff.

# CUSTOMIZABLE: Collect/Prepare data
datapoint_size = 1000
batch_size = 1000
steps = 10000
actual_b = 2
feature_number = 5
learning_rate = 0.001
log_file = "graphs"

# Fake data
all_xs, all_ys = [], []
actual_W = np.random.randint(low=-10, high=10, size=feature_number)

for i in range(datapoint_size):
    # Create fake data for y = actual_W1 * x1 + actual_W2 * x2  + ... + actual_WN * xN + actual_b
    a = i % 10
    xx = np.random.randint(low=-a-10, high=a+10, size=feature_number)
    yy = np.matmul(a=xx, b=actual_W) + actual_b
    all_xs.append(xx)
    all_ys.append(yy)
all_xs = np.array(all_xs)
all_ys = np.transpose([all_ys])

# Model construction
# placeholder for input variable
x = tf.placeholder(dtype=tf.float32, shape=[None, feature_number], name='x')

# definition of weight matrix and bias vector
W = tf.Variable(initial_value=tf.zeros([feature_number, 1]), name='W')
b = tf.Variable(initial_value=tf.zeros([1]), name='b')

# definition of outcome variable ( y = W1 * x1 + W2 * x2 + b )
with tf.name_scope("model"):
    product = tf.matmul(a=x, b=W)
    y_pred = tf.add(x=product, y=b, name='prediction')
# Add summary ops to collect data
tf.summary.histogram("weights", W)
tf.summary.histogram("biases", b)
tf.summary.histogram("y", y_pred)

# placeholder for true outcome values
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_true')

# definition for cost function (loss function ...) sum((yt -yp)^2)
with tf.name_scope("cost"):
    cost = tf.reduce_mean(input_tensor=tf.square(x=(y_true - y_pred)), name='cost_func')
    tf.summary.scalar("cost", cost)

# definition for gradient decent object with step == learning_rate and cost as minimization target
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=cost)

# for performance reasons and other things Tensorflow has to be executed within session
sess = tf.Session()

# Merge all the summaries and write them out to /tmp/logs
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_file, sess.graph)

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
    if i % 10 == 0:
        all_feed = {x: all_xs, y_true: all_ys}
        result = sess.run(fetches=merged, feed_dict=all_feed)
        writer.add_summary(summary=result, global_step=i)
    else:
        feed = {x: xs, y_true: ys}
        sess.run(fetches=train_step, feed_dict=feed)
        print("Iteration {iter} and cost is {cost} ".format(iter=i, cost=sess.run(fetches=cost, feed_dict=feed)))

print("Actual: W={W} and  b={b}".format(W=actual_W, b=actual_b))
print("Predicted: W={W} and  b={b}".format(W=sess.run(fetches=W), b=sess.run(fetches=b)))
