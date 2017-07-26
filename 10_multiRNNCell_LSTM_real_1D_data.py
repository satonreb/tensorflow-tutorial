import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


INPUT_SEQUENCE_LENGTH = 7
OUTPUT_SEQUENCE_LENGTH = 1
OUTPUT_SEQUENCE_STEPS_AHEAD = 2
TRAIN_SPLIT = 2
BATCH_SIZE = 50
N_ITERATIONS = 50000
LSTM_1_N = 16
HIDDEN_FC_N = 512
LEARNING_RATE = 1e-4
DATA_PATH = "data/euro-foreign-exchange-reference-.csv"

df = pd.read_csv(DATA_PATH)
df = df[:-3]
df['Date'] = df['Date'].map(lambda st: pd.datetime.strptime(st, '%Y-%m-%d'))

X = np.array(df['Date'])
y = np.array(df['Euro foreign exchange reference rates'])

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
# a = np.array([[[1], [2], [3], [4], [5]], [[21], [22], [23], [24], [25]],[[41], [42], [43], [44], [45]]])
# a.shape
# y_input_train_mod.shape
# ======================================================================================================================

# Define model
with tf.name_scope('INPUT'):
    # Input accepts arbitrary length sequence as input variable
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH], name='X')
    # x = tf.placeholder(dtype=tf.float32, shape=[None, None], name='x')
    # Input is of size [BATCH_COUNT, SEQUENCE_LENGTH, N_CLASSES]
    inputs = tf.expand_dims(input=x, axis=2)
    variable_summaries(inputs)
# Define RNN bit
with tf.name_scope('RNN'):
    lstm_1 = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_1_N)
    lstm_2 = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_1_N * 2)
    lstm_2 = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_1_N * 4)
    lstm_2 = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_1_N * 8)
    cells = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_1, lstm_2])
    rnn_output, _ = tf.nn.dynamic_rnn(cell=cells, inputs=inputs, dtype=tf.float32)
    # rnn_dropout = tf.nn.dropout(x=rnn_output, keep_prob=0.5)

# Standard Dense bit
with tf.name_scope('DNN'):
    output = tf.transpose(a=rnn_output, perm=[1, 0, 2])
    hidden_fc = tf.layers.dense(inputs=output[-1], units=HIDDEN_FC_N, activation=tf.nn.relu)
    # fc_dropout = tf.nn.dropout(x=hidden_fc, keep_prob=0.1)

with tf.name_scope('LR'):
    y_pred = tf.layers.dense(inputs=hidden_fc, units=OUTPUT_SEQUENCE_LENGTH)
    variable_summaries(y_pred)
# ======================================================================================================================
# Define loss
with tf.name_scope('LOSS'):
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SEQUENCE_LENGTH], name='TRUTH')
    # y_true = tf.placeholder(dtype=tf.float32, shape=[None, None], name='truth')
    loss = tf.reduce_mean(input_tensor=tf.square(x=(y_pred - y_true)))
    train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)
    variable_summaries(loss)
# ======================================================================================================================
# Train model
sess = tf.InteractiveSession()
sess.run(fetches=tf.global_variables_initializer())

log_file = "graphs/lstm_simple/multi_lstm"
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_file, sess.graph)

input_size = len(y_input_train_mod)
for s in range(N_ITERATIONS):
    ind_n = np.random.choice(a=input_size, size=BATCH_SIZE, replace=False)
    x_batch = y_input_train_mod[ind_n]
    y_batch = y_output_train[ind_n]

    feed_dict = {x: x_batch, y_true: y_batch}
    sess.run(fetches=train_step, feed_dict=feed_dict)

    if s % 1000 == 0:
        train_loss = loss.eval(feed_dict={x: y_input_train_mod, y_true: y_output_train})
        val_loss = loss.eval(feed_dict={x: y_input_test_mod, y_true: y_output_test})

        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e} ".format(e=s, steps=N_ITERATIONS,
                                                                          tr_e=train_loss, ts_e=val_loss)
        print(msg)

        summary = merged.eval(feed_dict={x: y_input_test_mod, y_true: y_output_test})
        writer.add_summary(summary, s)

# ======================================================================================================================
y_predictions_train = y_pred.eval(feed_dict={x: y_input_train_mod})
y_predictions_test = y_pred.eval(feed_dict={x: y_input_test_mod})
# ======================================================================================================================
marker_size = 3
plt.plot(x_input_train, y_input_train,
         color='black',
         label='train-input')

plt.plot(x_output_train, y_output_train,
         color='red',
         label='train-output')

plt.plot(x_output_train, y_predictions_train,
         color='blue',
         label='train-prediction')

plt.plot(x_input_test, y_input_test,
         color='cyan',
         label='test-input')

plt.plot(x_output_test, y_output_test,
         color='green',
         label='test-output')

plt.plot(x_output_test, y_predictions_test,
         color='orange',
         label='test-prediction')

plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
# ======================================================================================================================
