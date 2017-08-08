import numpy as np
import pandas as pd
import tensorflow as tf

pd.options.display.max_rows = 10
pd.options.display.width = 1000

# ======================================================================================================================
# Resets the graph
tf.reset_default_graph()
# ======================================================================================================================


def dif_date_to_int(dataframe, col_name):
    dataframe[col_name] = dataframe[col_name].fillna(value=0.0).astype('timedelta64[D]').astype(int)
    return dataframe


def create_numpy_arrays(y_dict, x_dict, labels_list):
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


# Data preparation =====================================================================================================

data_path = 'data/ophtha_0601_grpd_vars_v2.csv'
config_path = 'data/ophtha_0601_grpd_vars_v2var_config_edited.csv'

conf = pd.read_csv(filepath_or_buffer=config_path)
date_col = list(conf['Column'].loc[conf['Column'].str.contains('date')])

df = pd.read_csv(filepath_or_buffer=data_path, na_values='NA', low_memory=False,
                 parse_dates=date_col, infer_datetime_format=True)

data = df.loc[:, ['patient_id', 'eye', 'va_12_date', 'va_12']].drop_duplicates()
data['key'] = data['patient_id'].astype(str).str.cat(others=data['eye'])
data = (data.drop(labels=['patient_id', 'eye'], axis=1)
        .sort_values(by=['key', 'va_12_date'])
        .groupby('key')
        .filter(lambda b: len(b) > 1)
        .loc[:, ['va_12_date', 'key', 'va_12']])
data['date_diff'] = data.groupby('key')['va_12_date'].diff()


# ======================================================================================================================
uinq_df = data.loc[:, ['key']].drop_duplicates().assign(id=lambda x: range(0, len(x)))
data = pd.merge(left=data, right=uinq_df, on='key').drop(labels=['key', 'va_12_date'], axis=1)
rec_len = data.groupby('id').size()
data = dif_date_to_int(dataframe=data, col_name='date_diff')

def get_records(data, id):
    q = data[data['id'] == id].drop(labels='id', axis=1)
    for i in range(1, len(q)):
        w = q.iloc[:i+1, :]
        hist = w.iloc[:-1, :].values
        e = w.iloc[-1, :]
        target = e['va_12']
        prompt = e['date_diff']
        return hist, target, prompt, i


for id in range(0, len(data['id'].unique())):
    record = get_records(data=data, id=id)
    for i in range(0, rec_len.loc[id]):
        next(record)








# ======================================================================================================================



max_val = data.loc[:, ['va_12', 'date_diff']].max()

labels = data.loc[data['rank'] == 1, :].drop(labels=['va_12_date', 'rank'], axis=1)
features = data.loc[data['rank'] > 1, :].drop(labels=['va_12_date', 'rank'], axis=1)
labels = dif_date_to_int(dataframe=labels, col_name='date_diff')
features = dif_date_to_int(dataframe=features, col_name='date_diff')

max_delta_time = int(max_val[1].days)
max_va_12 = max_val[0]

labels['date_diff'] = labels['date_diff'] / max_delta_time
labels['va_12'] = labels['va_12'] / max_va_12
features['date_diff'] = features['date_diff'] / max_delta_time
features['va_12'] = features['va_12'] / max_va_12

l_dict = {k: v.values for k, v in labels.set_index(keys='key', append=True).groupby(level='key')}
f_dict = {k: v.values for k, v in features.set_index(keys='key', append=True).groupby(level='key')}

s_dict = {k: v.shape[0] for k, v in f_dict.items()}
s_df = pd.DataFrame.from_dict(data=s_dict, orient='index')
max_seq = int(s_df.max())

f_dict_pad = {k: np.concatenate((np.zeros(shape=(max_seq - v.shape[0], v.shape[1])), v)) for k, v in f_dict.items()}

labels_keys = labels['key'].tolist()
test_labels = np.random.choice(a=labels_keys, size=int(0.2 * len(labels_keys)), replace=False).tolist()
train_val_labels = list(set(labels_keys).difference(set(test_labels)))
val_labels = np.random.choice(a=train_val_labels, size=int(0.2 * len(train_val_labels)), replace=False).tolist()
train_labels = list(set(train_val_labels).difference(set(val_labels)))

x_train, y_train = create_numpy_arrays(y_dict=l_dict, x_dict=f_dict_pad, labels_list=train_labels)
x_val, y_val = create_numpy_arrays(y_dict=l_dict, x_dict=f_dict_pad, labels_list=val_labels)
x_test, y_test = create_numpy_arrays(y_dict=l_dict, x_dict=f_dict_pad, labels_list=test_labels)

# Neural Network model =================================================================================================
FEATURES = y_train.shape[2]
SEQ_LEN = max_seq
LSTM_1_N = 512
OUTPUT_SEQUENCE_LENGTH = y_train.shape[1]
INITIAL_LEARNING_RATE = 1e-2
LEARNING_RATE_DECAY_STEPS = 5000
LEARNING_RATE_DECAY_RATE = 0.96
LOG_FILE = 'graph'
BATCH_SIZE = 100
INPUT_SIZE = y_train.shape[0]
EPOCHS = 1000
N_ITERATIONS = EPOCHS * (INPUT_SIZE // BATCH_SIZE)
RNN_DEPTH = 4

with tf.variable_scope('setup'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, SEQ_LEN, FEATURES], name='x')
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_SEQUENCE_LENGTH, FEATURES], name='TRUTH')
    with tf.name_scope('learning_rate'):
        global_step = tf.Variable(initial_value=0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE, global_step=global_step,
                                                   decay_steps=LEARNING_RATE_DECAY_STEPS,
                                                   decay_rate=LEARNING_RATE_DECAY_RATE, staircase=True)
    variable_summaries(learning_rate)

# Define RNN bit
with tf.name_scope('RNN'):
    lstm_cells = []
    for nu in range(0, RNN_DEPTH):
        lstm = tf.nn.rnn_cell.LSTMCell(num_units=LSTM_1_N // 2 ** nu)
        lstm_cells.append(lstm)
    cells = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cells)
    rnn_output, _ = tf.nn.dynamic_rnn(cell=cells, inputs=x, dtype=tf.float32)

# Define Dense bit
with tf.name_scope('LR'):
    output = tf.transpose(a=rnn_output, perm=[1, 0, 2])
    y_pred = tf.layers.dense(inputs=output[-1], units=FEATURES)
    prediction = tf.expand_dims(input=y_pred, axis=1, name='PREDICTION')
    variable_summaries(prediction)

# ======================================================================================================================
# Define loss
with tf.name_scope('LOSS'):
    loss = tf.reduce_mean(input_tensor=tf.square(x=tf.subtract(x=prediction, y=y_true)))
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss, global_step=global_step)
    variable_summaries(loss)

# ======================================================================================================================
# Train model
sess = tf.InteractiveSession()
sess.run(fetches=tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

for s in range(N_ITERATIONS):
    ind_n = np.random.choice(a=INPUT_SIZE, size=BATCH_SIZE, replace=False)
    x_batch = x_train[ind_n]
    y_batch = y_train[ind_n]

    feed_dict = {x: x_batch, y_true: y_batch}
    sess.run(fetches=train_step, feed_dict=feed_dict)

    if s % 1000 == 0:
        train_loss = loss.eval(feed_dict={x: x_batch, y_true: y_batch})
        val_loss = loss.eval(feed_dict={x: x_val, y_true: y_val})

        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e} ".format(e=s, steps=N_ITERATIONS,
                                                                          tr_e=train_loss, ts_e=val_loss)
        print(msg)

        summary = merged.eval(feed_dict={x: x_val, y_true: y_val})
        writer.add_summary(summary, s)

# ======================================================================================================================
y_predictions_train = y_pred.eval(feed_dict={x: x_train})
y_predictions_val = y_pred.eval(feed_dict={x: x_val})
# ======================================================================================================================
# Test
test_loss = loss.eval(feed_dict={x: x_test, y_true: y_test})
print("Test loss is: {loss}".format(loss=test_loss))
y_predictions_test = y_pred.eval(feed_dict={x: x_test})
# ======================================================================================================================
