import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit

# ======================================================================================================================
# Resets the graph
tf.reset_default_graph()

# ======================================================================================================================
# Data generation


def data_split(data, input_seq_len, output_seq_len, output_seq_steps_ahead):
    steps_ahead = output_seq_steps_ahead - 1
    seq_number = len(data) + 1 - steps_ahead - input_seq_len - output_seq_len
    data_input = np.array([data[index: index + input_seq_len] for index in range(seq_number)])
    data_output = np.array([
        data[index + input_seq_len + steps_ahead: index + input_seq_len + steps_ahead + output_seq_len]
        for index in range(seq_number)])
    return data_input, data_output


POINT_NUMBER = 6000
INPUT_SEQUENCE_LENGTH = 10
OUTPUT_SEQUENCE_LENGTH = 3
OUTPUT_SEQUENCE_STEPS_AHEAD = 1
TRAIN_SPLIT = 2
BATCH_SIZE = 70
N_ITERATIONS = 10000
LSTM_NEURONS = 50
LEARNING_RATE = 1e-3

COEF = 10 * np.pi
X = np.linspace(start=-COEF, stop=COEF, num=POINT_NUMBER)
y = X * np.sin(X)

x_input, x_output = data_split(data=X,
                               input_seq_len=INPUT_SEQUENCE_LENGTH,
                               output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                               output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

y_input, y_output = data_split(data=y,
                               input_seq_len=INPUT_SEQUENCE_LENGTH,
                               output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                               output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

data_split = TimeSeriesSplit(n_splits=TRAIN_SPLIT)
for train_index, test_index in data_split.split(y_input):
    x_input_train, x_input_test = x_input[train_index], x_input[test_index]
    y_input_train, y_input_test = y_input[train_index], y_input[test_index]
    x_output_train, x_output_test = x_output[train_index], x_output[test_index]
    y_output_train, y_output_test = y_output[train_index], y_output[test_index]

y_input_train_mod = y_input_train
y_input_test_mod = y_input_test

# ======================================================================================================================
# Define model
with tf.name_scope('inputs'):
    # Input accepts arbitrary length sequence as input variable
    x = tf.placeholder(dtype=tf.float32, shape=[None, None], name='x')
    # Input is of size [BATCH_COUNT, SEQUENCE_LENGTH, N_CLASSES]
    inputs = tf.expand_dims(input=x, axis=2, name='x_dim_expand')

# ======================================================================================================================

# Define Encoder LSTM (RNN) bit
with tf.name_scope('encoder'):
    encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_NEURONS)
    # encoder_lstm_output, _ = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=inputs, dtype=tf.float32)

# Define Decoder LSTM (RNN) bit
with tf.name_scope('decoder'):
    decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_NEURONS)

    cells = tf.nn.rnn_cell.MultiRNNCell(cells=[encoder_cell, decoder_cell])

    rnn_output, _ = tf.nn.dynamic_rnn(cell=cells, inputs=inputs, dtype=tf.float32)

# Standard Dense bit
with tf.name_scope('FC'):
    output = tf.transpose(a=rnn_output, perm=[1, 0, 2])
    y_pred = tf.layers.dense(inputs=output[-1], units=OUTPUT_SEQUENCE_LENGTH)

# ======================================================================================================================
# Define loss
with tf.name_scope('loss'):
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SEQUENCE_LENGTH], name='truth')
    cost = tf.reduce_mean(input_tensor=tf.square(x=(y_pred - y_true)))
    tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=cost)

# ======================================================================================================================
# Run model
sess = tf.InteractiveSession()
sess.run(fetches=tf.global_variables_initializer())

log_file = "graphs/lstm_simple/dynamic"
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_file, sess.graph)

input_size = len(y_input_train_mod)
for s in range(N_ITERATIONS):
    ind_n = np.random.choice(a=input_size, size=BATCH_SIZE, replace=False)
    x_batch = y_input_train_mod[ind_n]
    y_batch = y_output_train[ind_n]

    feed_dict = {x: x_batch, y_true: y_batch}
    sess.run(fetches=train_step, feed_dict=feed_dict)

    if s % 100 == 0:
        train_loss = cost.eval(feed_dict={x: y_input_train_mod, y_true: y_output_train})
        val_loss = cost.eval(feed_dict={x: y_input_test_mod, y_true: y_output_test})

        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e} ".format(e=s, steps=N_ITERATIONS,
                                                                          tr_e=train_loss, ts_e=val_loss)
        print(msg)

        summary = merged.eval(feed_dict={x: y_input_test_mod, y_true: y_output_test})
        writer.add_summary(summary, s)

y_predictions_train = y_pred.eval(feed_dict={x: y_input_train_mod})
y_predictions_test = y_pred.eval(feed_dict={x: y_input_test_mod})

marker_size = 3
plt.scatter(x_input_train, y_input_train,
            color='black',
            label='train-input',
            s=marker_size * 10)

plt.scatter(x_output_train, y_output_train,
            color='red',
            label='train-output',
            s=marker_size * 5)

plt.scatter(x_output_train, y_predictions_train,
            color='blue',
            label='train-prediction',
            s=marker_size)

plt.scatter(x_input_test, y_input_test,
            color='cyan',
            label='test-input',
            s=marker_size * 10)

plt.scatter(x_output_test, y_output_test,
            color='green',
            label='test-output',
            s=marker_size * 5)

plt.scatter(x_output_test, y_predictions_test,
            color='orange',
            label='test-prediction',
            s=marker_size)

plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
