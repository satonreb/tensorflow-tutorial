import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# Data generation
def gen_data(data, train_test_split, sequence_length):
    sequence_length = sequence_length + 1
    result = []
    for index in range(len(data) - sequence_length+1):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    np.random.shuffle(result)
    x = result[:, :-1]
    y = result[:, -1]

    tscv = TimeSeriesSplit(n_splits=train_test_split)

    for train_index, test_index in tscv.split(result):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return {'train': x_train, 'test': x_test}, {'train': y_train, 'test': y_test}

POINT_NUMBER = 10 * 4
TRAIN_SPLIT = 3
SEQ_LEN = 10

x = np.linspace(start=0, stop=2*np.pi, num=POINT_NUMBER)
y = np.sin(x)

X, Y = gen_data(data=y, train_test_split=TRAIN_SPLIT, sequence_length=SEQ_LEN)

x_train = X['train']
y_train = Y['train']
x_test = X['test']
y_test = Y['test']









input_dim = SEQ_LEN
output_dim = 1

# Define model
x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim, 1], name='x')
y_true = tf.placeholder(dtype=tf.float32, shape=[None, output_dim], name='truth')

# LSTM (RNN) bit :
x_seq = tf.unstack(value=x, num=input_dim, axis=1)

cell_1 = tf.nn.rnn_cell.LSTMCell(num_units=50)

lstm_keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='lstm_keep_prob')

cell_1_drop = tf.contrib.rnn.DropoutWrapper(cell_1, output_keep_prob=lstm_keep_prob)
val_1, state_1 = tf.nn.static_rnn(cell=cell_1_drop, inputs=x_seq, dtype=tf.float32)
last = val_1[-1]

# Standart Dense bit:

input_layer_size = 50
W = tf.Variable(initial_value=tf.truncated_normal(shape=[input_layer_size, output_dim], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[output_dim]))
y_pred = tf.matmul(a=last, b=W) + b

# Define loss
LEARNING_RATE = 1e-4
cost = tf.reduce_mean(input_tensor=tf.square(x=(y_pred - y_true)))
train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=cost)

# Running
EPOCH_SIZE = len(x_train)
EPOCHS = 1024 * 16
BATCH_SIZE = 256
STEPS = int(EPOCH_SIZE * EPOCHS / BATCH_SIZE)
PRINT_INVERSE_FREQ = 50

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print("Starting TensorFlow implementation")
for s in range(STEPS):
    if s % PRINT_INVERSE_FREQ == 0:
        train_loss = cost.eval(feed_dict={x: x_train,
                                          y_true: y_train,
                                          lstm_keep_prob: 1})

        val_loss = cost.eval(feed_dict={x: x_train,
                                        y_true: y_train,
                                        lstm_keep_prob: 1})

        msg = "step: {e}/{steps}, loss: {tr_e}, val_loss: {ts_e}".format(e=s, tr_e=train_loss, ts_e=val_loss, steps=STEPS)
        print(msg)

    n = len(x_train)
    ind_n = np.random.choice(n, BATCH_SIZE, replace=False)
    x_batch = list(np.array(x_train)[ind_n])
    y_batch = list(np.array(y_train)[ind_n])

    feed_dict = {x: x_batch, y_true: y_batch, lstm_keep_prob: lstm_keep_prob}

    sess.run(train_step, feed_dict=feed_dict)