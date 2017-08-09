import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


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


def data_split(data, input_seq_len, output_seq_len, output_seq_steps_ahead):
    """Splits data into input and output datasets in accordance with supplied parameters."""
    steps_ahead = output_seq_steps_ahead - 1
    seq_number = len(data) + 1 - steps_ahead - input_seq_len - output_seq_len
    data_input = np.array([data[index: index + input_seq_len] for index in range(seq_number)])
    data_output = np.array(list(
        data[index + input_seq_len + steps_ahead: index + input_seq_len + steps_ahead + output_seq_len]
        for index in range(seq_number)))
    return data_input, data_output


def prep_data(array, input_seq_len, output_seq_len, output_seq_steps_ahead, expand_dim=True, expand_dim_axis=2):
    """Prepares data in accordance with supplied parameters by splitting and expanding datasets into third [:,:,1]
    dimension."""
    data = data_split(data=array, input_seq_len=input_seq_len, output_seq_len=output_seq_len,
                      output_seq_steps_ahead=output_seq_steps_ahead)
    if expand_dim:
        data = list(map(lambda a: np.expand_dims(a, axis=expand_dim_axis), data))
    return np.array(data[0]), np.array(data[1])


# ======================================================================================================================
# Resets the graph
tf.reset_default_graph()
# ======================================================================================================================
# Hyperparameters
INPUT_SEQUENCE_LENGTH = 7
OUTPUT_SEQUENCE_LENGTH = 1
OUTPUT_SEQUENCE_STEPS_AHEAD = 4
N_SPLITS = 2
BATCH_SIZE = 70
N_ITERATIONS = 30000
LSTM_1_N = 16
INITIAL_LEARNING_RATE = 1e-3
LEARNING_RATE_DECAY_STEPS = 5000
LEARNING_RATE_DECAY_RATE = 0.96
FC_1_N = 20
log_file = "graphs/lstm"
# ======================================================================================================================
# Real Wold Data
DATA_PATH = "data/data.csv"
df = pd.read_csv(DATA_PATH)
df['Date'] = df['Date'].map(lambda st: pd.datetime.strptime(st, '%Y-%m-%d'))
df['GBP/EUR'] = pd.to_numeric(df['GBP/EUR'], errors='coerce')
df.dropna(inplace=True)
# These bits have changed
today = np.datetime64('today').astype('datetime64[ns]').astype('uint64')
X = np.array(df['Date']).astype('uint64') / today
Y1 = df['GBP/EUR'].values
# ======================================================================================================================
# Synthetic Data
# X = np.linspace(start=-5 * np.pi, stop=10 * np.pi, num=500)
# Y1 = np.sin(X) / 2 - np.sin(-X * 2) + X / 10
Y2 = np.sin(X*100)
# X = X / (10 * np.pi)
# plt.plot(X, Y2)
# ======================================================================================================================
x_input, x_output = prep_data(array=X, input_seq_len=INPUT_SEQUENCE_LENGTH, output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                              output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD, expand_dim=False, expand_dim_axis=2)
y1_input, y1_output = prep_data(array=Y1, input_seq_len=INPUT_SEQUENCE_LENGTH, output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD, expand_dim=True, expand_dim_axis=2)
y2_input, y2_output = prep_data(array=Y2, input_seq_len=INPUT_SEQUENCE_LENGTH, output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                                output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD, expand_dim=True, expand_dim_axis=2)
# **********************************************************************************************************************
# These bits have changed
input = np.concatenate((y1_input, y2_input), axis=2)
output = np.concatenate((y1_output, y2_output), axis=2)
# **********************************************************************************************************************

# Split data into Training and Test datasets
for train_val_index, test_index in TimeSeriesSplit(n_splits=N_SPLITS).split(input):
    y_input_test_val, y_input_train = input[train_val_index], input[test_index]
    x_input_test_val, x_input_train = x_input[train_val_index], x_input[test_index]
    y_output_test_val, y_output_train = output[train_val_index], output[test_index]
    x_output_test_val, x_output_train = x_output[train_val_index], x_output[test_index]


# Split Test data into Train and Validation datasets
for train_index, val_index in TimeSeriesSplit(n_splits=N_SPLITS).split(x_input_test_val):
    y_input_test, y_input_val = y_input_test_val[train_index], y_input_test_val[val_index]
    x_input_test, x_input_val = x_input_test_val[train_index], x_input_test_val[val_index]
    y_output_test, y_output_val = y_output_test_val[train_index], y_output_test_val[val_index]
    x_output_test, x_output_val = x_output_test_val[train_index], x_output_test_val[val_index]

# ======================================================================================================================
# Define model
with tf.name_scope('SETUP'):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE, global_step=global_step,
                                               decay_steps=LEARNING_RATE_DECAY_STEPS,
                                               decay_rate=LEARNING_RATE_DECAY_RATE, staircase=True)
    # Input is of shape [BATCH_COUNT, SEQUENCE_LENGTH, FEATURES]
    FEATURES = y_input_train.shape[2]
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH, FEATURES], name='X')
    # Input accepts arbitrary length sequence as input variable
    # x = tf.placeholder(dtype=tf.float32, shape=[None, None, FEATURES], name='x')
    variable_summaries(x)

# Define RNN bit
with tf.name_scope('RNN'):
    lstm_1 = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_1_N)
    lstm_2 = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_1_N // 2)
    cells = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_1, lstm_2])
    rnn_output, _ = tf.nn.dynamic_rnn(cell=cells, inputs=x, dtype=tf.float32)

# Define Dense bit
with tf.name_scope('DNN'):
    output = tf.transpose(a=rnn_output, perm=[1, 0, 2])
    last_rnn = output[-1]
    # last = tf.gather(output, int(output.get_shape()[0]) - 1)
    fc_1 = tf.layers.dense(inputs=last_rnn, units=FC_1_N, activation=tf.nn.relu)
    variable_summaries(fc_1)

with tf.name_scope('LR'):
    y_pred = tf.layers.dense(inputs=fc_1, units=FEATURES)
    prediction = tf.expand_dims(input=y_pred, axis=1, name='PREDICTION')
    variable_summaries(y_pred)

# ======================================================================================================================
# Define loss
with tf.name_scope('LOSS'):
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1, FEATURES], name='TRUTH')
    # y_true = tf.placeholder(dtype=tf.float32, shape=[None, None], name='truth')
    mse = tf.losses.mean_squared_error(labels=y_true, predictions=prediction)
    loss = tf.sqrt(x=mse, name='rmse')
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
    variable_summaries(loss)

# ======================================================================================================================
# Train model
sess = tf.InteractiveSession()
sess.run(fetches=tf.global_variables_initializer())

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
        val_loss = loss.eval(feed_dict={x: y_input_val, y_true: y_output_val})

        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e} ".format(e=s, steps=N_ITERATIONS,
                                                                          tr_e=train_loss, ts_e=val_loss)
        print(msg)

        summary = merged.eval(feed_dict={x: y_input_val, y_true: y_output_val})
        writer.add_summary(summary, s)

# ======================================================================================================================
y_predictions_train = prediction.eval(feed_dict={x: y_input_train})
y_predictions_val = prediction.eval(feed_dict={x: y_input_val})
# ======================================================================================================================
marker_size = 3
plt.figure()
plt.plot(x_input_train, y_input_train[:, :, 0],
         color='black',
         label='train-input')

plt.plot(x_input_train, y_predictions_train[:, :, 0],
         color='orange',
         label='train-prediction')

plt.plot(x_output_val, y_input_val[:, :, 0],
         color='blue',
         label='val-input')

plt.plot(x_output_val, y_predictions_val[:, :, 0],
         color='red',
         label='val-prediction')

plt.xlabel("x")
plt.ylabel("y_1")
plt.grid()
plt.legend()
# ======================================================================================================================
marker_size = 3
plt.figure()
plt.plot(x_input_train, y_input_train[:, :, 1],
         color='black',
         label='train-input')

plt.plot(x_input_train, y_predictions_train[:, :, 1],
         color='orange',
         label='train-prediction')

plt.plot(x_output_val, y_input_val[:, :, 1],
         color='blue',
         label='val-input')

plt.plot(x_output_val, y_predictions_val[:, :, 1],
         color='red',
         label='val-prediction')

plt.xlabel("x")
plt.ylabel("y_0")
plt.grid()
plt.legend()
# ======================================================================================================================
# Test
test_loss = loss.eval(feed_dict={x: y_input_test, y_true: y_output_test})
print("Test loss is: {loss}".format(loss=test_loss))
y_predictions_test = prediction.eval(feed_dict={x: y_input_test})
# ======================================================================================================================
plt.figure()
plt.plot(x_input_test, y_input_test[:, :, 0],
         color='black',
         label='test-input')

plt.plot(x_output_test, y_predictions_test[:, :, 0],
         color='orange',
         label='test-prediction')
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()
# ======================================================================================================================
plt.figure()
plt.plot(x_input_test, y_input_test[:, :, 1],
         color='black',
         label='test-input')

plt.plot(x_output_test, y_predictions_test[:, :, 1],
         color='orange',
         label='test-prediction')
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()
# ======================================================================================================================
