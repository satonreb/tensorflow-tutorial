import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Script shows how to use TensorFlow for logistic regression for multiple features and classes,
# and use batches for the training.

# CUSTOMIZABLE: Collect/Prepare data
batch_size = 100
steps = 1000
feature_number = 784  # 28x28 pixel images mapped to (784 x 1) vector
learning_rate = 0.5
log_file = "graphs"
class_count = 10  # MNIST dataset has 10 classes that can be predicted

# MNIST data
data_dir = "/tmp/tensorflow/mnist/input_data"
mnist = input_data.read_data_sets(data_dir, one_hot=True)

# Model construction
# placeholder for input variable
x = tf.placeholder(dtype=tf.float32, shape=[None, feature_number], name="x")

# definition of weight matrix and bias vector
W = tf.Variable(initial_value=tf.zeros([feature_number, class_count]), name="W")
b = tf.Variable(initial_value=tf.zeros([class_count]), name="b")

# definition of outcome variable ( y = W1 * x1 + W2 * x2 + ... + WN *xN + b)
with tf.name_scope("model"):
    y_pred = tf.add(x=tf.matmul(a=x, b=W), y=b, name="prediction")

# placeholder for true outcome values
y_true = tf.placeholder(dtype=tf.float32, shape=[None, class_count], name="y_true")

# definition for cost function (loss function ...) sum((yt -yp)^2)
with tf.name_scope("cost"):
    # cost = tf.reduce_mean(input_tensor=tf.square(x=(y_true - y_pred)), name="cost_func")
    cross_entropy = tf.reduce_mean(input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred),
                                   name="cost_func")

# definition for gradient decent object with step == learning_rate and cost as minimization target
with tf.name_scope("train"):
    # train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=cost)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# for performance reasons and other things Tensorflow has to be executed within session
sess = tf.Session()

# define variable initialisation
init = tf.global_variables_initializer()

# perform actual variable initialisation
sess.run(fetches=init)

# training steps
for _ in range(steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    feed = {x: batch_xs, y_true: batch_ys}
    sess.run(fetches=train_step, feed_dict=feed)


# Test trained model
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_true: mnist.test.labels}))
