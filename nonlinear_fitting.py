import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# CUSTOMIZABLE: Collect/Prepare data
X_RANGE_MIN, X_RANGE_MAX = -np.pi, np.pi
N = 1000
TEST_SIZE = 0.33
EPOCHS = 10000
LEARNING_RATE = 1e-3

# Create data
X = np.random.uniform(low=X_RANGE_MIN, high=X_RANGE_MAX, size=N)
noise = np.random.normal(scale=0.01, size=X.size)
y = (np.sin(X))+ noise

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
X_train = X_train.reshape(X_train.size, 1)
X_test = X_test.reshape(X_test.size, 1)
y_train = y_train.reshape(y_train.size, 1)
y_test = y_test.reshape(y_test.size, 1)

# Define model
input_layer_size = 1
output_layer_size = 1
x = tf.placeholder(tf.float32, shape=[None, input_layer_size])
y_true = tf.placeholder(tf.float32, shape=[None, output_layer_size])

# Linear
# W = tf.Variable(initial_value=tf.zeros([input_layer_size, output_layer_size]))
# b = tf.Variable(initial_value=tf.zeros([output_layer_size]))
# y_pred = tf.multiply(x, W) + b

# Dense NN
# First layer
n_neurons = 50
w1 = tf.Variable(initial_value=tf.truncated_normal(shape=[input_layer_size, n_neurons], stddev=0.1))
b1 = tf.Variable(initial_value=tf.constant(value=0.1, shape=[n_neurons]))
h1 = tf.nn.relu(tf.multiply(x=x, y=w1) + b1)
# Output layer
w2 = tf.Variable(initial_value=tf.truncated_normal(shape=[n_neurons, output_layer_size], stddev=0.1))
b2 = tf.Variable(initial_value=tf.constant(value=0.1, shape=[output_layer_size]))
y_pred = tf.matmul(a=h1, b=w2) + b2

# Define loss
cost = tf.reduce_mean(input_tensor=tf.square(x=(y_pred - y_true)))
train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=cost)

# Start the session
sess = tf.InteractiveSession()
sess.run(fetches=tf.global_variables_initializer())

# Train model
for i in range(EPOCHS):
    feed = {x: X_train, y_true: y_train}
    result = sess.run(fetches=train_step, feed_dict=feed)
    if i % 1000 == 0:
        train_error = cost.eval(feed_dict={x: X_train, y_true: y_train})
        test_error = cost.eval(feed_dict={x: X_test, y_true: y_test})
        print("epoch: {i}, train error: {train_error}, test error: {test_error}".format(
                i=i,
                train_error=train_error,
                test_error=test_error))

# Plot results
plt.scatter(X_test, y_test,  color='b')
xx = np.sort(X_test, axis=0)
yy = y_pred.eval(feed_dict={x: xx})
plt.plot(xx, yy, color='r', linewidth=3)
plt.xlabel("X")
plt.ylabel("y")
plt.show()
