import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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
INPUT_SEQUENCE_LENGTH = 28
OUTPUT_SEQUENCE_LENGTH = 1
OUTPUT_SEQUENCE_STEPS_AHEAD = 14
N_SPLITS = 4
BATCH_SIZE = 70
N_ITERATIONS = 10000
LSTM_1_N = 16
LEARNING_RATE = 1e-3
# ======================================================================================================================
# # Real Wold Data
# DATA_PATH = "data/euro-foreign-exchange-reference-.csv"
# df = pd.read_csv(DATA_PATH)
# df = df[:-3]
# df['Date'] = df['Date'].map(lambda st: pd.datetime.strptime(st, '%Y-%m-%d'))
# # **********************************************************************************************************************
# # These bits have changed
# today = np.datetime64('today').astype('datetime64[ns]').astype('uint64')
# X = np.array(df['Date']).astype('uint64') / today
# # **********************************************************************************************************************
# Y = np.array(df['Euro foreign exchange reference rates'])
# ======================================================================================================================
# Synthetic Data
X = np.linspace(start=-5 * np.pi, stop=10 * np.pi, num=500)
Y = np.sin(X) / 2 - np.sin(-X * 5) + X

# plt.plot(X, Y)
# ======================================================================================================================

t_input, x_output = prep_data(array=X, input_seq_len=INPUT_SEQUENCE_LENGTH, output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                              output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD, expand_dim=True, expand_dim_axis=2)
x_input, y_output = prep_data(array=Y, input_seq_len=INPUT_SEQUENCE_LENGTH, output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                              output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD, expand_dim=True, expand_dim_axis=2)

# Split data into Training and Test datasets
for train_val_index, test_index in TimeSeriesSplit(n_splits=N_SPLITS).split(x_input):
    t_input_train_val, t_input_test = t_input[train_val_index], t_input[test_index]
    x_input_train_val, x_input_test = x_input[train_val_index], x_input[test_index]
    x_output_train_val, x_output_test = x_output[train_val_index], x_output[test_index]
    y_output_train_val, y_output_test = y_output[train_val_index], y_output[test_index]

# Split Test data into Train and Validation datasets
for train_index, val_index in TimeSeriesSplit(n_splits=N_SPLITS).split(x_input_train_val):
    t_input_train, t_input_val = t_input_train_val[train_index], t_input_train_val[val_index]
    x_input_train, x_input_val = x_input_train_val[train_index], x_input_train_val[val_index]
    x_output_train, x_output_val = x_output_train_val[train_index], x_output_train_val[val_index]
    y_output_train, y_output_val = y_output_train_val[train_index], y_output_train_val[val_index]

# ======================================================================================================================
# Define model

with tf.name_scope('SETUP'):
    # Input is of shape [BATCH_COUNT, SEQUENCE_LENGTH, FEATURES]
    FEATURES = x_input_train.shape[2]
    x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH, FEATURES], name='X')
    t = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH, FEATURES], name='T')
    # Input accepts arbitrary length sequence as input variable
    # x = tf.placeholder(dtype=tf.float32, shape=[None, None, FEATURES], name='x')
    variable_summaries(x)

# Define RNN bit
with tf.name_scope('RNN'):
    cell = tf.contrib.rnn.PhasedLSTMCell(num_units=LSTM_1_N)
    rnn_output, _ = tf.nn.dynamic_rnn(cell=cell, inputs=(x, t), dtype=tf.float32)
# **********************************************************************************************************************
# Define Dense bit
with tf.name_scope('LR'):
    output = tf.transpose(a=rnn_output, perm=[1, 0, 2])
    y_pred = tf.layers.dense(inputs=output[-1], units=OUTPUT_SEQUENCE_LENGTH)
    prediction = tf.expand_dims(input=y_pred, axis=2, name='PREDICTION')
    variable_summaries(y_pred)

# ======================================================================================================================
# Define loss
with tf.name_scope('LOSS'):
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SEQUENCE_LENGTH, 1], name='TRUTH')
    # y_true = tf.placeholder(dtype=tf.float32, shape=[None, None, 1], name='truth')
    loss = tf.reduce_mean(input_tensor=tf.square(x=tf.subtract(x=y_pred, y=y_true)))
    train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=loss)
    variable_summaries(loss)

# ======================================================================================================================
# Train model
sess = tf.InteractiveSession()
sess.run(fetches=tf.global_variables_initializer())

log_file = "graphs/lstm"
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_file, sess.graph)

input_size = len(x_input_train)
for s in range(N_ITERATIONS):
    ind_n = np.random.choice(a=input_size, size=BATCH_SIZE, replace=False)
    t_batch = t_input_train[ind_n]
    x_batch = x_input_train[ind_n]
    y_batch = y_output_train[ind_n]

    feed_dict = {t: t_batch, x: x_batch, y_true: y_batch}
    sess.run(fetches=train_step, feed_dict=feed_dict)

    if s % 1000 == 0:
        train_loss = loss.eval(feed_dict={t: t_input_train, x: x_input_train, y_true: y_output_train})
        val_loss = loss.eval(feed_dict={t: t_input_val, x: x_input_val, y_true: y_output_val})

        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e} ".format(e=s, steps=N_ITERATIONS,
                                                                          tr_e=train_loss, ts_e=val_loss)
        print(msg)

        summary = merged.eval(feed_dict={t: t_input_val, x: x_input_val, y_true: y_output_val})
        writer.add_summary(summary, s)

# ======================================================================================================================
y_predictions_train = y_pred.eval(feed_dict={t: t_input_train, x: x_input_train})
y_predictions_val = y_pred.eval(feed_dict={t: t_input_val, x: x_input_val})
# ======================================================================================================================
marker_size = 3
plt.figure()
plt.plot(t_input_train[:, :, 0], x_input_train[:, :, 0],
         color='black',
         label='train-input')

plt.plot(x_output_train.flatten(), y_predictions_train.flatten(),
         color='orange',
         label='train-prediction')

plt.plot(t_input_val[:, :, 0], x_input_val[:, :, 0],
         color='blue',
         label='val-input')

plt.plot(x_output_val.flatten(), y_predictions_val.flatten(),
         color='red',
         label='val-prediction')

plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
# ======================================================================================================================
# Test
test_loss = loss.eval(feed_dict={t: t_input_test, x: x_input_test, y_true: y_output_test})
print("Test loss is: {loss}".format(loss=test_loss))
y_predictions_test = y_pred.eval(feed_dict={t: t_input_test, x: x_input_test})
plt.figure()
plt.plot(t_input_test[:, :, 0], x_input_test[:, :, 0],
         color='black',
         label='test-input')

plt.plot(x_output_test.flatten(), y_predictions_test.flatten(),
         color='green',
         label='test-prediction')
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()
# ======================================================================================================================
