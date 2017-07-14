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

POINT_NUMBER = 1000
PAST_SEQUENCE_LENGTH = 60
FUTURE_SEQUENCE_LENGTH = 1
FUTURE_SEQUENCE_STEPS_AHEAD = 1
TRAIN_SPLIT = 3
BATCH_SIZE = 30
N_ITERATIONS = 5000

# x = np.linspace(start=0, stop=4*np.pi, num=POINT_NUMBER)
COEF = 10*np.pi
NUM = int(POINT_NUMBER/2)
X = np.concatenate((-COEF*np.sort(np.random.random(size=NUM)), COEF*np.sort(np.random.random(size=NUM))))
y = np.sin(X)

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

input_dim = PAST_SEQUENCE_LENGTH
output_dim = FUTURE_SEQUENCE_LENGTH
hidden_size = 50

# Define model
x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim, 1], name='x')
y_true = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name='truth')

# LSTM (RNN) bit :
x_seq = tf.unstack(value=x, num=input_dim, axis=1)
cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
output, out_state = tf.nn.static_rnn(cell=cell, inputs=x_seq, dtype=tf.float32)


# Standard Dense bit:
W = tf.Variable(initial_value=tf.truncated_normal(shape=[hidden_size, output_dim], stddev=0.1), name='weight')
b = tf.Variable(initial_value=tf.constant(0.1, shape=[output_dim]), name='bias')
y_pred = tf.add(x=tf.matmul(a=output[-1], b=W), y=b, name='prediction')

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

y_future_sequences_train_predictions_tf = y_pred.eval(feed_dict={x: y_input_train})
y_future_sequences_test_predictions_tf = y_pred.eval(feed_dict={x: y_input_test})


marker_size = 3
plt.scatter(x_future_train.flatten(), y_future_train.flatten(),
            color='black',
            label='training',
            s=marker_size)
plt.scatter(x_future_test.flatten(), y_future_test.flatten(),
            color='red',
            label='test',
            s=marker_size)
plt.scatter(x_future_train.flatten(), y_future_sequences_train_predictions_tf.flatten(),
            color='cyan',
            label='training - predicted (TensorFlow)',
            s=marker_size)
plt.scatter(x_future_test.flatten(), y_future_sequences_test_predictions_tf.flatten(),
            color='orange',
            label='test - predicted (TensorFlow)',
            s=marker_size)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
