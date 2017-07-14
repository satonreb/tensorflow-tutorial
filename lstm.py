import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


def data_split(data, past_seq_len,  future_seq_len, future_seq_steps_ahead):
    steps_ahead = future_seq_steps_ahead-1
    seq_number = len(data) + 1 - steps_ahead - past_seq_len - future_seq_len
    data_past = np.array([data[index: index + past_seq_len] for index in range(seq_number)])
    data_future = np.array([
        data[index + past_seq_len + steps_ahead: index + past_seq_len + steps_ahead + future_seq_len]
        for index in range(seq_number)])
    return data_past, data_future

POINT_NUMBER = 20
PAST_SEQUENCE_LENGTH = 5
FUTURE_SEQUENCE_LENGTH = 2
FUTURE_SEQUENCE_STEPS_AHEAD = 1
TRAIN_SPLIT = 3

x = np.linspace(start=0, stop=2*np.pi, num=POINT_NUMBER)
y = np.sin(x)

x_past, x_future = data_split(data=x,
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
    y_futuregit s_train, y_future_test = y_future[train_index], y_future[test_index]


a = y_past_train.reshape(y_past_train.shape, 1)

# y_test_output = list(np.reshape(y_future_sequences_test, tuple(list(y_future_sequences_test.shape))))



#
# input_dim = SEQ_LEN
# output_dim = 1
#
# # Define model
# x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim, 1], name='x')
# y_true = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name='truth')
#
# # LSTM (RNN) bit :
# x_seq = tf.unstack(value=x, num=input_dim, axis=1)
#
# cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=50)
#
# lstm_keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='lstm_keep_prob')
#
# cell_1_drop = tf.contrib.rnn.DropoutWrapper(cell_1, output_keep_prob=lstm_keep_prob)
# val_1, state_1 = tf.nn.static_rnn(cell=cell_1_drop, inputs=x_seq, dtype=tf.float32)
# last = val_1[-1]
#
# # Standart Dense bit:
#
# input_layer_size = 50
# W = tf.Variable(initial_value=tf.truncated_normal(shape=[input_layer_size, output_dim], stddev=0.1))
# b = tf.Variable(tf.constant(0.1, shape=[output_dim]))
# y_pred = tf.matmul(a=last, b=W) + b
#
# # Define loss
# LEARNING_RATE = 1e-4
# cost = tf.reduce_mean(input_tensor=tf.square(x=(y_pred - y_true)))
# train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=cost)
#
# # Running
# EPOCH_SIZE = len(x_train)
# EPOCHS = 1024 * 16
# BATCH_SIZE = 256
# STEPS = int(EPOCH_SIZE * EPOCHS / BATCH_SIZE)
# PRINT_INVERSE_FREQ = 50
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# print("Starting TensorFlow implementation")
# for s in range(STEPS):
#     if s % PRINT_INVERSE_FREQ == 0:
#         train_loss = cost.eval(feed_dict={x: x_train,
#                                           y_true: y_train,
#                                           lstm_keep_prob: 1})
#
#         val_loss = cost.eval(feed_dict={x: x_train,
#                                         y_true: y_train,
#                                         lstm_keep_prob: 1})
#
#         msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e}".format(e=s, tr_e=train_loss, ts_e=val_loss, steps=STEPS)
#         print(msg)
#
#     n = len(x_train)
#     ind_n = np.random.choice(n, BATCH_SIZE, replace=False)
#     x_batch = list(np.array(x_train)[ind_n])
#     y_batch = list(np.array(y_train)[ind_n])
#
#     feed_dict = {x: x_batch, y_true: y_batch, lstm_keep_prob: lstm_keep_prob}
#
#     sess.run(train_step, feed_dict=feed_dict)