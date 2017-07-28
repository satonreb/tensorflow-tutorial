import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ======================================================================================================================
# Resets the graph
tf.reset_default_graph()
# ======================================================================================================================
# Hyperparameters
TEST_SIZE = 0.33
LEARNING_RATE = 1e-2
BATCH_SIZE = 120
N_ITERATIONS = 10000
# ======================================================================================================================
# Synthetic Data
X = np.linspace(start=-5, stop=10, num=2000)
noise = np.random.normal(scale=1, size=X.size)
Y = 2.0 * X + 3.0 + noise

# plt.plot(X, Y)
# ======================================================================================================================
# Split data
data = list(map(lambda a: np.expand_dims(a, axis=1), [X, Y]))
x_train_val, x_test, y_train_val, y_test = train_test_split(data[0], data[1], test_size=TEST_SIZE)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=TEST_SIZE)

print('Shape of Training dataset is: {train}, Validation: {val}, Test: {test}'.format(train=x_train.shape,
                                                                                      val=x_val.shape,
                                                                                      test=x_test.shape))
# ======================================================================================================================
# Define model

# Input is of shape [BATCH_COUNT, FEATURES]
FEATURES = x_train.shape[1]
x = tf.placeholder(tf.float32, shape=[None, FEATURES], name='X')

# Define Dense bit
with tf.variable_scope('LR'):
    prediction = tf.layers.dense(inputs=x, units=FEATURES, name='PREDICTION')

# Define loss
y_true = tf.placeholder(dtype=tf.float32, shape=[None, FEATURES], name='TRUTH')
loss = tf.reduce_mean(input_tensor=tf.square(x=tf.subtract(x=prediction, y=y_true)))
train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)

# ======================================================================================================================
# Train model
sess = tf.InteractiveSession()
sess.run(fetches=tf.global_variables_initializer())

input_size = len(y_train)
for s in range(N_ITERATIONS):
    ind_n = np.random.choice(a=input_size, size=BATCH_SIZE, replace=False)
    x_batch = x_train[ind_n]
    y_batch = y_train[ind_n]

    feed_dict = {x: x_batch, y_true: y_batch}
    sess.run(fetches=train_step, feed_dict=feed_dict)

    if s % 1000 == 0:
        train_loss = loss.eval(feed_dict={x: x_train, y_true: y_train})
        val_loss = loss.eval(feed_dict={x: x_val, y_true: y_val})

        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e} ".format(e=s, steps=N_ITERATIONS,
                                                                          tr_e=train_loss, ts_e=val_loss)
        print(msg)

# ======================================================================================================================
y_predictions_train, w, b = prediction.eval(feed_dict={x: x_train})

y_predictions_val = prediction.eval(feed_dict={x: x_val})


# ======================================================================================================================
def flatten_sort(sort_by, b):
    if len(sort_by.shape) > 0:
        sort_by = sort_by.flatten()
    ind = np.argsort(sort_by)
    return np.array([sort_by.flatten()[i] for i in ind]), np.array([b.flatten()[i] for i in ind])


marker_size = 3
plt.figure()

plt.scatter(x_train, y_train,
            color='red',
            label='train-true',
            s=marker_size)

input_train, output_train = flatten_sort(sort_by=x_train, b=y_predictions_train)
plt.plot(input_train, output_train,
         color='orange',
         label='train-predictions',
         linewidth=4)

plt.scatter(x_val, y_val,
            color='blue',
            label='val-true',
            s=marker_size)

input_val, output_val = flatten_sort(sort_by=x_val, b=y_predictions_val)
plt.plot(input_val, output_val,
         color='black',
         label='val-predictions')

plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
# ======================================================================================================================
# Test
test_loss = loss.eval(feed_dict={x: x_test, y_true: y_test})
print("Test loss is: {loss}".format(loss=test_loss))
y_predictions_test = prediction.eval(feed_dict={x: x_test})
plt.figure()

plt.scatter(x_test, y_test,
            color='black',
            label='test-true',
            s=marker_size)

input_test, output_test = flatten_sort(sort_by=x_test, b=y_predictions_test)
plt.plot(input_test, output_test,
         color='orange',
         label='test-predictions')

plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()
# ======================================================================================================================
