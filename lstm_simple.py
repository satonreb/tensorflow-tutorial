import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


# Data generation
def data_split(data, past_seq_len,  future_seq_len, future_seq_steps_ahead):
    steps_ahead = future_seq_steps_ahead-1
    seq_number = len(data) + 1 - steps_ahead - past_seq_len - future_seq_len
    data_past = np.array([data[index: index + past_seq_len] for index in range(seq_number)])
    data_future = np.array([
        data[index + past_seq_len + steps_ahead: index + past_seq_len + steps_ahead + future_seq_len]
        for index in range(seq_number)])
    return data_past, data_future

POINT_NUMBER = 200
PAST_SEQUENCE_LENGTH = 5
FUTURE_SEQUENCE_LENGTH = 1
FUTURE_SEQUENCE_STEPS_AHEAD = 10
TRAIN_SPLIT = 2
BATCH_SIZE = 6
N_ITERATIONS = 2000
LSTM_NEURONS = 2

X = np.linspace(start=-8*np.pi, stop=8*np.pi, num=POINT_NUMBER)
y = X*np.sin(X)

x_past, x_future = data_split(data=X,
                              past_seq_len=PAST_SEQUENCE_LENGTH,
                              future_seq_len=FUTURE_SEQUENCE_LENGTH,
                              future_seq_steps_ahead=FUTURE_SEQUENCE_STEPS_AHEAD)

y_past, y_future = data_split(data=y,
                              past_seq_len=PAST_SEQUENCE_LENGTH,
                              future_seq_len=FUTURE_SEQUENCE_LENGTH,
                              future_seq_steps_ahead=FUTURE_SEQUENCE_STEPS_AHEAD)

tscv = TimeSeriesSplit(n_splits=TRAIN_SPLIT)
for train_index, test_index in tscv.split(y_past):
    x_past_train, x_past_test = x_past[train_index], x_past[test_index]
    y_past_train, y_past_test = y_past[train_index], y_past[test_index]
    x_future_train, x_future_test = x_future[train_index], x_future[test_index]
    y_future_train, y_future_test = y_future[train_index], y_future[test_index]

y_input_train = np.expand_dims(a=y_past_train, axis=2)
y_output_train = y_future_train

y_input_test = np.expand_dims(a=y_past_test, axis=2)
y_output_test = y_future_test

# TODO: What are shape parameters in x? [batch_count, sequance_lenght, ??????] ?
# TODO: Understand end explain LSTM bit: particularly, why have to use unstack? how to use static_rnn?
# TODO: how to substitute it with dynamic_rnn? Why nothing works as expected?
# TODO: Look into batch creation and evaluation of various things during computation time

# Define model
x = tf.placeholder(dtype=tf.float32, shape=[None, PAST_SEQUENCE_LENGTH, 1], name='x')
y_true = tf.placeholder(dtype=tf.float32, shape=[None, FUTURE_SEQUENCE_LENGTH], name='truth')

# LSTM (RNN) bit :
cell = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_NEURONS)
output, _ = tf.nn.dynamic_rnn(cell=cell, inputs=x, dtype=tf.float32)


# Standard Dense bit:
# We use output[-1] here as we are only interested the output after whole seance has been looked at.
# verison 1
# W = tf.Variable(initial_value=tf.truncated_normal(shape=[hidden_size, output_dim], stddev=0.1), name='weight')
# b = tf.Variable(initial_value=tf.constant(0.1, shape=[output_dim]), name='bias')
# y_pred = tf.add(x=tf.matmul(a=output[-1], b=W), y=b, name='prediction')
# verison 2
# y_pred = tf.layers.dense(inputs=output[-1], units=output_dim)
# verison 3
y_pred = tf.contrib.layers.fully_connected(inputs=output[-1], num_outputs=FUTURE_SEQUENCE_LENGTH, activation_fn=None)
# Quenstion: Why would you need at least three ways to do same thing?

# Define loss
LEARNING_RATE = 1e-4
cost = tf.reduce_mean(input_tensor=tf.square(x=(y_pred - y_true)))
train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=cost)

# Run model
sess = tf.InteractiveSession()
sess.run(fetches=tf.global_variables_initializer())

input_size = len(y_input_train)
for s in range(N_ITERATIONS):
    ind_n = np.random.choice(a=input_size, size=BATCH_SIZE, replace=False)
    x_batch = y_input_train[ind_n]
    y_batch = y_output_train[ind_n]

    if s % 100 == 0:
        train_loss = cost.eval(feed_dict={x: y_input_train,
                                          y_true: y_output_train})
        val_loss = cost.eval(feed_dict={x: y_input_test,
                                        y_true: y_output_test})
        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e}".format(e=s,
                                                                         tr_e=train_loss,
                                                                         ts_e=val_loss,
                                                                         steps=N_ITERATIONS)
        print(msg)
    feed_dict = {x: x_batch, y_true: y_batch}
    sess.run(fetches=train_step, feed_dict=feed_dict)

y_predictions_train = y_pred.eval(feed_dict={x: y_input_train})
y_predictions_test = y_pred.eval(feed_dict={x: y_input_test})

marker_size = 3
plt.scatter(x_past_train, y_past_train,
            color='black',
            label='training',
            s=marker_size)
plt.scatter(x_past_test, y_past_test,
            color='red',
            label='test',
            s=marker_size)

plt.scatter(x_future_train, y_predictions_train,
            color='cyan',
            label='training - predicted',
            s=marker_size)
plt.scatter(x_future_test.flatten(), y_predictions_test.flatten(),
            color='orange',
            label='test - predicted',
            s=marker_size)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()