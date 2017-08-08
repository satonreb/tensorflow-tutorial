import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

pd.options.display.max_rows = 10
pd.options.display.width = 1000

# ======================================================================================================================
# Resets the graph
tf.reset_default_graph()
# ======================================================================================================================
"""
Scripts uses only patients historic va_12 measurements to predict va_12 value on a particular date.

Model: 
Model consists of multiple RNN (LSTM) Cells stacked together which take in variable length sequences of patient historic
va_12 measurements and dates of the measurements. Last relevant output of RNN is then concatenated with the date 
for which va_12 prediction will be made. This then is passed to a dense layer that is connected to another dens layer 
with one neuron that outputs prediction. 

Loss:
Loss is calculated as root mean square error (RMSE).

Optimisation:
Script uses Adam optimizer with exponentially decaying learning rate.
"""


# ======================================================================================================================


def dif_date_to_int(dataframe, col_name):
    """ Function substates zeros for NAT  dates and converts timedelta day format to integers in dataframe for a given
    column."""
    dataframe[col_name] = dataframe[col_name].fillna(value=0.0).astype('timedelta64[D]').astype(int)
    return dataframe


def create_numpy_arrays(y_dict, x_dict, labels_list):
    """Function creates two arrays from given dictionary of values when given list of keys."""
    yy = np.array(list(map(lambda a: y_dict[a], labels_list)))
    xx = np.array(list(map(lambda a: x_dict[a], labels_list)))
    return xx, yy


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


# Data location ========================================================================================================
data_dir = 'data'
path_to_pickle = os.path.join(data_dir, '01_prepared_data.pkl')
data_path = os.path.join(data_dir, 'ophtha_0601_grpd_vars_v2.csv')
config_path = os.path.join(data_dir, 'ophtha_0601_grpd_vars_v2var_config_edited.csv')

# Data preparation  or Data Retrieval ==================================================================================
if os.path.exists(path_to_pickle) is not True:
    # Reads in a given configuration and data CSV files
    conf = pd.read_csv(filepath_or_buffer=config_path)
    date_col = list(conf['Column'].loc[conf['Column'].str.contains('date')])

    df = pd.read_csv(filepath_or_buffer=data_path, na_values='NA', low_memory=False,
                     parse_dates=date_col, infer_datetime_format=True)
    # Selects 'patient_id', 'eye', 'va_12_date', 'va_12' columns and removes duplicates
    data = df.loc[:, ['patient_id', 'eye', 'va_12_date', 'va_12']].drop_duplicates()
    # Creates identifier 'key' column combining patient ids and eye columns 
    data['key'] = data['patient_id'].astype(str).str.cat(others=data['eye'])
    # Removes 'patient_id' and 'eye' and removes entries that contain only one key value
    data = (data.drop(labels=['patient_id', 'eye'], axis=1)
            .sort_values(by=['key', 'va_12_date'])
            .groupby('key')
            .filter(lambda b: len(b) > 1)
            .loc[:, ['va_12_date', 'key', 'va_12']])
    # Calculates days passed between measurements for each kay value
    data['date_diff'] = data.groupby('key')['va_12_date'].diff()
    # Creates rank column that shows measurement date order in descending order, the last 
    # measurement date is showen as 1.0 
    data['rank'] = data.groupby('key')['va_12_date'].rank(ascending=False)
    # Finds largest values for the measurement and passed time 
    max_val = data.loc[:, ['va_12', 'date_diff']].max()
    max_delta_time = int(max_val[1].days)
    max_va_12 = max_val[0]
    # Splits data into target values and features, where targets are measurement whihc were taken last and features are 
    # all remaining records for particular key 
    targets = data.loc[data['rank'] == 1, :].drop(labels=['va_12_date', 'rank'], axis=1)
    targets = dif_date_to_int(dataframe=targets, col_name='date_diff')
    features = data.loc[data['rank'] > 1, :].drop(labels=['va_12_date', 'rank'], axis=1)
    features = dif_date_to_int(dataframe=features, col_name='date_diff')
    # Scale all records by devising each entry by the maximum value in the records 
    targets['date_diff'] = targets['date_diff'] / max_delta_time
    targets['va_12'] = targets['va_12'] / max_va_12
    features['date_diff'] = features['date_diff'] / max_delta_time
    features['va_12'] = features['va_12'] / max_va_12
    # Create a dictionary of scaled target and feature values
    target_dict = {k: v.values for k, v in targets.set_index(keys='key', append=True).groupby(level='key')}
    feature_dict = {k: v.values for k, v in features.set_index(keys='key', append=True).groupby(level='key')}
    # Creates a dictionary of the record length for each entry and finds longest entry
    s_dict = {k: v.shape[0] for k, v in feature_dict.items()}
    s_df = pd.DataFrame.from_dict(data=s_dict, orient='index')
    max_seq = int(s_df.max())
    # Pads all feature records by zeros to longest entry length, [1,2,3] -> [1,2,3,0,0,0,0]
    feature_dict_pad = {k: np.concatenate((v, np.zeros(shape=(max_seq - v.shape[0], v.shape[1]))))
                        for k, v in feature_dict.items()}
    # Collects all key values
    labels_keys = targets['key'].tolist()
    # Randomly selects test, train and validation labels
    labels_test = np.random.choice(a=labels_keys, size=int(0.2 * len(labels_keys)), replace=False).tolist()
    train_labels_val = list(set(labels_keys).difference(set(labels_test)))
    labels_val = np.random.choice(a=train_labels_val, size=int(0.2 * len(train_labels_val)), replace=False).tolist()
    labels_train = list(set(train_labels_val).difference(set(labels_val)))
    # Creates feature (x) and target(y) arrays using test, train and validation labels
    x_train, y_train = create_numpy_arrays(y_dict=target_dict, x_dict=feature_dict_pad, labels_list=labels_train)
    x_val, y_val = create_numpy_arrays(y_dict=target_dict, x_dict=feature_dict_pad, labels_list=labels_val)
    x_test, y_test = create_numpy_arrays(y_dict=target_dict, x_dict=feature_dict_pad, labels_list=labels_test)
    # Creates an array of original record lengths for test, train and validation datasets
    seq_len_train = np.array(list(map(lambda a: s_dict[a], labels_train)))
    seq_len_val = np.array(list(map(lambda a: s_dict[a], labels_val)))
    seq_len_test = np.array(list(map(lambda a: s_dict[a], labels_test)))
    # creates dictionary of all reinvent data
    pickle_dict = {'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test,
                   'y_test': y_test, 'seq_len_train': seq_len_train, 'seq_len_val': seq_len_val,
                   'seq_len_test': seq_len_test, 'max_seq': max_seq, 'max_delta_time': max_delta_time,
                   'max_va_12': max_va_12}
    # Write data dict to a file
    output = open(path_to_pickle, 'wb')
    pickle.dump(pickle_dict, output)
    output.close()
else:
    # Read data dict back from the file
    pkl_file = open(path_to_pickle, 'rb')
    pickle_dict = pickle.load(pkl_file)
    pkl_file.close()
    x_train = pickle_dict['x_train']
    y_train = pickle_dict['y_train']
    x_val = pickle_dict['x_val']
    y_val = pickle_dict['y_val']
    x_test = pickle_dict['x_test']
    y_test = pickle_dict['y_test']
    seq_len_train = pickle_dict['seq_len_train']
    seq_len_val = pickle_dict['seq_len_val']
    seq_len_test = pickle_dict['seq_len_test']
    max_seq = pickle_dict['max_seq']
    max_delta_time = pickle_dict['max_delta_time']
    max_va_12 = pickle_dict['max_va_12']

# ======================================================================================================================
# Splits target vectors to date and value array, as we are going to tray to learn to predict values on the certain date
y_date_train = y_train[:, :, 1]
y_target_train = y_train[:, :, 0]
y_date_val = y_val[:, :, 1]
y_target_val = y_val[:, :, 0]
y_date_test = y_test[:, :, 1]
y_target_test = y_test[:, :, 0]

# Parameters ===========================================================================================================
# Hyperparameters
LSTM_1_N = 256
INITIAL_LEARNING_RATE = 1e-2
LEARNING_RATE_DECAY_STEPS = 5000
LEARNING_RATE_DECAY_RATE = 0.96
RNN_DEPTH = 3
EPOCHS = 50
BATCH_SIZE = 70
DNN_N = 128
# Additional parameters
INPUT_SIZE = y_train.shape[0]
FEATURES = x_train.shape[2]
OUTPUT_SEQUENCE_LENGTH = y_target_train.shape[1]
MAX_SEQ_LEN = max_seq
LOG_FILE = 'graph'
N_ITERATIONS = EPOCHS * (INPUT_SIZE // BATCH_SIZE)

# Neural Network model =================================================================================================
with tf.variable_scope('setup'):
    # sequence_length is a placeholder which allows us supply length of each sequence in the batch
    sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')
    # x is placeholder for features
    x = tf.placeholder(dtype=tf.float32, shape=[None, MAX_SEQ_LEN, FEATURES], name='x')
    # date is a placeholder for date input for which we want to make a prediction
    date = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='date')
    # true value on the given date
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_true')
    with tf.name_scope('learning_rate'):
        # define decaying learning rate with given decay steps and rate
        global_step = tf.Variable(initial_value=0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE, global_step=global_step,
                                                   decay_steps=LEARNING_RATE_DECAY_STEPS,
                                                   decay_rate=LEARNING_RATE_DECAY_RATE, staircase=True,
                                                   name='learning_rate')
    # record learning_rate statistics
    variable_summaries(learning_rate)

# Define RNN bit
with tf.name_scope('RNN'):
    # given RNN_DEPTH value loop create LSTM cells where number of unit in each layer decays
    # as 2**layer_depth (1, 2, 4, 8, ...)
    def lstm_cell(units):
        return tf.nn.rnn_cell.LSTMCell(num_units=units)


    cells = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell(units=LSTM_1_N // 2 ** i) for i in range(RNN_DEPTH)])
    rnn_output, _ = tf.nn.dynamic_rnn(cell=cells, inputs=x, dtype=tf.float32, sequence_length=sequence_length)

# Define Dense bit
with tf.name_scope('DNN'):
    # This is true only if all sequences lengths are equal and not padded.
    # output = tf.transpose(a=rnn_output, perm=[1, 0, 2])
    # concat = tf.concat([output[-1], date], axis=1, name='concat')

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e, if a sequence length is 10, we need
    # to retrieve the 10th output

    last_rnn_output = tf.gather_nd(rnn_output, tf.stack(values=[tf.range(tf.shape(rnn_output)[0]), sequence_length - 1],
                                                        axis=1))
    concat = tf.concat(values=[last_rnn_output, date], axis=1)

    # tf.nn.dynamic_rnn state contains outputs c and h where h is equal to the last relevant output
    # Combine RNN output with the date value for which prediction is made
    # concat = tf.concat(values=[rnn_states.h, date], axis=1)

    # Create dens layer with given neron count and ReLu activation function
    dnn = tf.layers.dense(inputs=concat, units=DNN_N, activation=tf.nn.relu)

with tf.name_scope('linear_regression'):
    # Define additional dense layer that performs linear regression and record its output
    prediction = tf.layers.dense(inputs=dnn, units=1)
    variable_summaries(prediction)

# Define loss
with tf.name_scope('loss'):
    # Define loss function as root square mean (RMSE) and record its value
    loss = tf.reduce_mean(input_tensor=tf.square(x=tf.subtract(x=prediction, y=y_true)))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
    variable_summaries(loss)

# Train model ==========================================================================================================
sess = tf.InteractiveSession()
sess.run(fetches=tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

counter = INPUT_SIZE // BATCH_SIZE
index_list = range(0, INPUT_SIZE)
batch_indexes = (index_list[i:i + BATCH_SIZE] for i in range(0, INPUT_SIZE, BATCH_SIZE))
for s in range(N_ITERATIONS):
    inx = next(batch_indexes)
    if s == counter:
        counter += INPUT_SIZE // BATCH_SIZE
        index_list = np.random.permutation(index_list)
        batch_indexes = (index_list[i:i + BATCH_SIZE] for i in range(0, INPUT_SIZE, BATCH_SIZE))

    x_batch = x_train[inx]
    y_date_batch = y_date_train[inx]
    y_target_batch = y_target_train[inx]
    seq_len_batch = seq_len_train[inx]

    feed_dict = {x: x_batch, y_true: y_target_batch, sequence_length: seq_len_batch, date: y_date_batch}
    sess.run(fetches=train_step, feed_dict=feed_dict)

    if s % 100 == 0:
        train_loss = loss.eval(feed_dict={x: x_batch, y_true: y_target_batch,
                                          sequence_length: seq_len_batch, date: y_date_batch})

        val_loss = loss.eval(feed_dict={x: x_val, y_true: y_target_val, sequence_length: seq_len_val, date: y_date_val})
        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e} ".format(e=s, steps=N_ITERATIONS,
                                                                          tr_e=train_loss, ts_e=val_loss)
        print(msg)

        summary = merged.eval(
            feed_dict={x: x_val, y_true: y_target_val, sequence_length: seq_len_val, date: y_date_val})
        writer.add_summary(summary, s)

# ======================================================================================================================
# y_predictions_train = prediction.eval(feed_dict={x: x_train, sequence_length: seq_len_train, date: y_date_train})
# y_predictions_val = prediction.eval(feed_dict={x: x_val, sequence_length: seq_len_val, date: y_date_val})
# Test =================================================================================================================
test_loss = loss.eval(feed_dict={x: x_test, y_true: y_target_test, sequence_length: seq_len_test, date: y_date_test})
print("Test loss is: {loss}".format(loss=test_loss))
y_predictions_test = prediction.eval(feed_dict={x: x_test, sequence_length: seq_len_test, date: y_date_test})
# ======================================================================================================================
plt.figure()
plt.scatter(y_date_test, y_target_test, s=4, color='orange', label='ground truth')
plt.scatter(y_date_test, y_predictions_test, s=2, color='black', label='prediction')
plt.legend()
plt.xlabel(s='Days between measurements')
plt.ylabel(s='va_12')
plt.show()
# =====================================================================================================================
