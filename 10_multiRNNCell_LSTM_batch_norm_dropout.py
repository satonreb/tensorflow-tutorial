import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
INPUT_SEQUENCE_LENGTH = 14
OUTPUT_SEQUENCE_LENGTH = 3
OUTPUT_SEQUENCE_STEPS_AHEAD = 1
N_SPLITS = 4
BATCH_SIZE = 30
N_ITERATIONS = 20000
LSTM_1_N = 64
DNN_1_N = 32
INITIAL_LEARNING_RATE = 1e-2
LEARNING_RATE_DECAY_STEPS = 5000
LEARNING_RATE_DECAY_RATE = 0.96
L2_REG_BETA = 1e-5
log_file = "graphs/lstm"
# ======================================================================================================================
# # Real Wold Data
# DATA_PATH = "data/euro-foreign-exchange-reference-.csv"
# df = pd.read_csv(DATA_PATH)
# df = df[:-3]
# df['Date'] = df['Date'].map(lambda st: pd.datetime.strptime(st, '%Y-%m-%d'))
# X = np.array(df['Date'])
# Y = np.array(df['Euro foreign exchange reference rates'])
# ======================================================================================================================
# Synthetic Data
X = np.linspace(start=-5 * np.pi, stop=15 * np.pi, num=2000)
Y = (np.sin(X/2) - np.sin(-X*2) + np.cos(X*3) + X/3)/14
# plt.plot(X, Y)
# ======================================================================================================================

x_input, x_output = prep_data(array=X, input_seq_len=INPUT_SEQUENCE_LENGTH, output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                              output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD, expand_dim=False, expand_dim_axis=2)
y_input, y_output = prep_data(array=Y, input_seq_len=INPUT_SEQUENCE_LENGTH, output_seq_len=OUTPUT_SEQUENCE_LENGTH,
                              output_seq_steps_ahead=OUTPUT_SEQUENCE_STEPS_AHEAD, expand_dim=True, expand_dim_axis=2)

# Split data into Training and Test datasets
for train_val_index, test_index in TimeSeriesSplit(n_splits=N_SPLITS).split(y_input):
    x_input_train_val, x_input_test = x_input[train_val_index], x_input[test_index]
    y_input_train_val, y_input_test = y_input[train_val_index], y_input[test_index]
    x_output_train_val, x_output_test = x_output[train_val_index], x_output[test_index]
    y_output_train_val, y_output_test = y_output[train_val_index], y_output[test_index]

# Split Test data into Train and Validation datasets
for train_index, val_index in TimeSeriesSplit(n_splits=N_SPLITS).split(y_input_train_val):
    x_input_train, x_input_val = x_input_train_val[train_index], x_input_train_val[val_index]
    y_input_train, y_input_val = y_input_train_val[train_index], y_input_train_val[val_index]
    x_output_train, x_output_val = x_output_train_val[train_index], x_output_train_val[val_index]
    y_output_train, y_output_val = y_output_train_val[train_index], y_output_train_val[val_index]

# ======================================================================================================================
FEATURES = y_input_train.shape[2]
DNN_KEEP_PROB = 0.5
RNN_KEEP_PROB = 0.3
RNN_DEPTH = 3

# ======================================================================================================================
# Define model
with tf.name_scope('setup'):
    with tf.name_scope('variable_placeholders'):
        training = tf.placeholder(dtype=tf.bool, name='training_phase')
        rnn_keep_prob = tf.cond(pred=training,
                                true_fn=lambda: RNN_KEEP_PROB,
                                false_fn=lambda: 1.0,
                                name='RNN_dropout_keep_probability')

    with tf.name_scope('learning_rate'):
        global_step = tf.Variable(initial_value=0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE, global_step=global_step,
                                                   decay_steps=LEARNING_RATE_DECAY_STEPS,
                                                   decay_rate=LEARNING_RATE_DECAY_RATE, staircase=True)
        variable_summaries(learning_rate)

    with tf.name_scope('inputs'):
        # Input is of shape [BATCH_COUNT, SEQUENCE_LENGTH, FEATURES]
        x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_SEQUENCE_LENGTH, FEATURES], name='x')
        y_true = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SEQUENCE_LENGTH, 1], name='ground_truth')
        # Input accepts arbitrary length sequence as input variable
        # x = tf.placeholder(dtype=tf.float32, shape=[None, None, FEATURES], name='input')
        # y_true = tf.placeholder(dtype=tf.float32, shape=[None, None, 1], name='ground_truth')

# Define RNN bit
with tf.name_scope('RNN'):
    # cell = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_1_N)
    lstm_cells = []
    for nu in range(0, RNN_DEPTH):
        lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=LSTM_1_N // 2**nu)
        lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(cell=lstm, output_keep_prob=rnn_keep_prob)
        lstm_cells.append(lstm_dropout)
    cells = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cells)
    rnn_output, _ = tf.nn.dynamic_rnn(cell=cells, inputs=x, dtype=tf.float32)

# Define Dense bit
with tf.name_scope('DNN'):
    # Regularisation
    with tf.name_scope('l2_regulariser'):
        l2_regulariser = tf.contrib.layers.l2_regularizer(scale=L2_REG_BETA)

    output = tf.transpose(a=rnn_output, perm=[1, 0, 2])
    # last = tf.gather(val, int(val.get_shape()[0]) - 1)
    last = output[-1]
    dnn = tf.layers.dense(inputs=last, units=DNN_1_N, kernel_regularizer=l2_regulariser, activation=tf.nn.relu)
    dnn_output = tf.layers.dropout(inputs=dnn, rate=DNN_KEEP_PROB, training=training)

with tf.name_scope('prediciton'):
    y_pred = tf.layers.dense(inputs=dnn_output, units=OUTPUT_SEQUENCE_LENGTH)
    prediction = tf.expand_dims(input=y_pred, axis=2)
    variable_summaries(prediction)

# ======================================================================================================================
# Define loss
with tf.name_scope('loss'):
    reconstruction_loss = tf.reduce_mean(input_tensor=tf.square(x=tf.subtract(x=prediction, y=y_true)))
    reg_losses = tf.get_collection(key=tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n(inputs=[reconstruction_loss]+reg_losses)
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

    feed_dict = {x: x_batch, y_true: y_batch, training: True}
    sess.run(fetches=train_step, feed_dict=feed_dict)

    if s % 1000 == 0:
        train_loss = loss.eval(feed_dict={x: x_batch, y_true: y_batch, training: False})
        val_loss = loss.eval(feed_dict={x: y_input_val, y_true: y_output_val, training: False})

        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e} ".format(e=s, steps=N_ITERATIONS,
                                                                          tr_e=train_loss, ts_e=val_loss)
        print(msg)

        summary = merged.eval(feed_dict={x: y_input_val, y_true: y_output_val, training: False})
        writer.add_summary(summary, s)

# ======================================================================================================================
y_predictions_train = prediction.eval(feed_dict={x: y_input_train, training: False})
y_predictions_val = prediction.eval(feed_dict={x: y_input_val, training: False})
# ======================================================================================================================
marker_size = 10
plt.figure()
plt.plot(x_input_train, y_input_train[:, :, 0],
         color='black',
         label='train-input')

plt.scatter(x_output_train.flatten(), y_predictions_train.flatten(),
            color='orange',
            label='train-prediction',
            s=marker_size)

plt.plot(x_input_val, y_input_val[:, :, 0],
         color='blue',
         label='val-input')

plt.scatter(x_output_val.flatten(), y_predictions_val.flatten(),
            color='red',
            label='val-prediction',
            s=marker_size)

plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
# ======================================================================================================================
# Test
test_loss = loss.eval(feed_dict={x: y_input_test, y_true: y_output_test, training: False})
print("Test loss is: {loss}".format(loss=test_loss))
y_predictions_test = prediction.eval(feed_dict={x: y_input_test, training: False})
plt.figure()
a = 0

plt.plot(x_input_test[a], y_input_test[a, :, 0],
         color='black',
         label='test-input',
         marker='x')

plt.plot(x_output_test[a], y_output_test[a, :, 0],
         color='blue',
         label='test-truth',
         linewidth=4)

plt.plot(x_output_test[a], y_predictions_test[a, :, 0],
         color='green',
         label='test-prediction')

plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()
# ======================================================================================================================
