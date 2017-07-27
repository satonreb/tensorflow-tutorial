import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit

# ======================================================================================================================
# Resets the graph
tf.reset_default_graph()


# ======================================================================================================================


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


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


INPUT_SEQUENCE_LENGTH = 5
OUTPUT_SEQUENCE_LENGTH = 1
OUTPUT_SEQUENCE_STEPS_AHEAD = 10
N_SPLITS = 2
BATCH_SIZE = 70
N_ITERATIONS = 5000
learning_rate_decay_steps = 500
learning_rate_decay_rate = 0.96
LSTM_1_N = 4
FC_1_N = 10
INITIAL_LEARNING_RATE = 1e-2
L2_REG_BETA = 0.03
# DATA_PATH = "data/euro-foreign-exchange-reference-.csv"

# df = pd.read_csv(DATA_PATH)
# df = df[:-3]
# df['Date'] = df['Date'].map(lambda st: pd.datetime.strptime(st, '%Y-%m-%d'))
#
# X = np.array(df['Date'])
# Y = np.array(df['Euro foreign exchange reference rates'])

X = np.linspace(start=-5 * np.pi, stop=10 * np.pi, num=1000)
Y = 1 * np.sin(X) - 1

# plt.plot(X, Y)

x_input, x_output = data_split(data=X,
                               input_seq_len=INPUT_SEQUENCE_LENGTH,
                               output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                               output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

y_input, y_output = data_split(data=Y,
                               input_seq_len=INPUT_SEQUENCE_LENGTH,
                               output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                               output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD)

input = np.array([[[y_input[i, j], x_input[i, j]] for j in range(x_input.shape[1])] for i in range(x_input.shape[0])])

data_split = TimeSeriesSplit(n_splits=N_SPLITS)
for train_index, test_index in data_split.split(y_input):
    y_input_train, y_input_test = input[train_index], input[test_index]
    x_output_train, x_output_test = x_output[train_index], x_output[test_index]
    y_output_train, y_output_test = y_output[train_index], y_output[test_index]
# ======================================================================================================================
# a = np.array([[[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]], [[21, 2], [22, 2], [23,2 ], [24, 2], [25, 2]],[[41, 3], [42, 3], [43, 3], [44, 3], [45, 3]]])
# a.shape
# y_input_train.shape
# ======================================================================================================================
FEATURES = input.shape[2]
# Define model
with tf.name_scope('INPUT'):
    # Exponentialy reduce learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE, global_step=global_step,
                                               decay_steps=learning_rate_decay_steps,
                                               decay_rate=learning_rate_decay_rate, staircase=True)
    # Input accepts arbitrary length sequence as input variable
    # Input is of shape [BATCH_COUNT, SEQUENCE_LENGTH, FEATURES]
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH, FEATURES], name='X')
    # x = tf.placeholder(dtype=tf.float32, shape=[None, None, FEATURES], name='x')
    variable_summaries(x)

# Define RNN bit
with tf.name_scope('RNN'):
    lstm_1 = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_1_N)
    lstm_2 = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_1_N // 2)
    cells = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_1, lstm_2])
    rnn_output, _ = tf.nn.dynamic_rnn(cell=cells, inputs=x, dtype=tf.float32)

# Regularisation
with tf.name_scope('L2_REG'):
    l2_regulariser = tf.contrib.layers.l2_regularizer(scale=L2_REG_BETA)
    he_init = tf.contrib.layers.variance_scaling_initializer()  # He initialization

# Define Dense bit
with tf.name_scope('DNN'):
    output = tf.transpose(a=rnn_output, perm=[1, 0, 2])
    last_rnn = output[-1]
    # last = tf.gather(output, int(output.get_shape()[0]) - 1)
    fc_1 = tf.layers.dense(inputs=last_rnn, units=FC_1_N, activation=tf.nn.relu,
                           kernel_initializer=he_init, kernel_regularizer=l2_regulariser)
    variable_summaries(fc_1)

with tf.name_scope('LR'):
    y_pred = tf.layers.dense(inputs=fc_1, units=OUTPUT_SEQUENCE_LENGTH)
    variable_summaries(y_pred)

# ======================================================================================================================
# Define loss
with tf.name_scope('LOSS'):
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SEQUENCE_LENGTH], name='TRUTH')
    # y_true = tf.placeholder(dtype=tf.float32, shape=[None, None], name='truth')
    reconstruction_loss = tf.reduce_mean(input_tensor=tf.square(x=(y_pred - y_true)))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([reconstruction_loss] + reg_losses)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
    variable_summaries(loss)

# ======================================================================================================================
# Train model
sess = tf.InteractiveSession()
sess.run(fetches=tf.global_variables_initializer())

log_file = "graphs/lstm_simple/multi_lstm"
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_file, sess.graph)

input_size = len(y_input_train)
for s in range(N_ITERATIONS):
    ind_n = np.random.choice(a=input_size, size=BATCH_SIZE, replace=False)
    x_batch = y_input_train[ind_n]
    y_batch = y_output_train[ind_n]

    feed_dict = {x: x_batch, y_true: y_batch}
    sess.run(fetches=train_step, feed_dict=feed_dict)

    if s % 1000 == 0:
        train_loss = loss.eval(feed_dict={x: y_input_train, y_true: y_output_train})
        val_loss = loss.eval(feed_dict={x: y_input_test, y_true: y_output_test})

        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e} ".format(e=s, steps=N_ITERATIONS,
                                                                          tr_e=train_loss, ts_e=val_loss)
        print(msg)

        summary = merged.eval(feed_dict={x: y_input_test, y_true: y_output_test})
        writer.add_summary(summary, s)

# ======================================================================================================================
y_predictions_train = y_pred.eval(feed_dict={x: y_input_train})
y_predictions_test = y_pred.eval(feed_dict={x: y_input_test})
# ======================================================================================================================
marker_size = 3
plt.scatter(y_input_train[:, :, 1].flatten(), y_input_train[:, :, 0].flatten(),
            color='black',
            label='train-input',
            s=marker_size)

plt.plot(x_output_train.flatten(), y_predictions_train.flatten(),
         color='orange',
         label='train-prediction')

plt.scatter(y_input_test[:, :, 1].flatten(), y_input_test[:, :, 0].flatten(),
            color='blue',
            label='test-input',
            s=marker_size)

plt.plot(x_output_test.flatten(), y_predictions_test.flatten(),
         color='red',
         label='test-prediction')

plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
# ======================================================================================================================
