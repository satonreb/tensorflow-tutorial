import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Global config variables
num_steps = 10  # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 2
rnn_neurons = 4
learning_rate = 0.1
num_epochs = 1


def gen_data(size=1000000):
    x = np.array(np.random.choice(2, size=(size,)))
    y = []
    for i in range(size):
        threshold = 0.5
        if x[i - 3] == 1:
            threshold += 0.5
        if x[i - 8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            y.append(0)
        else:
            y.append(1)
    return x, np.array(y)

def gen_sin_data(size=1000000):
    x = np.linspace(start=-8 * np.pi, stop=8 * np.pi, num=size)
    y = x * np.sin(X)
    return x, np.array(y)


# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


"""
Placeholders
"""

x = tf.placeholder(tf.int32, [None, num_steps], name='input')
y_true = tf.placeholder(tf.int32, [None, num_steps], name='labels')

"""
Inputs
"""
rnn_inputs = tf.one_hot(x, num_classes)

"""
RNN
"""

cell = tf.contrib.rnn.BasicRNNCell(rnn_neurons)
rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_inputs, dtype=tf.float32)

"""
Predictions, loss, training step
"""

y_perd = tf.layers.dense(inputs=rnn_outputs, units=num_classes)

predictions = tf.nn.softmax(y_perd)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_perd)
total_loss = tf.reduce_mean(input_tensor=losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss=total_loss)

"""
Train the network
"""
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())
training_losses = []
for idx, epoch in enumerate(gen_epochs(n=num_epochs, num_steps=num_steps)):
    training_loss = 0
    print("\nEPOCH", idx)
    for step, (X, Y) in enumerate(epoch):
        training_loss_, _ = sess.run([total_loss, train_step], feed_dict={x: X, y_true: Y})
        training_loss += training_loss_
        if step % 250 == 0 and step > 0:
            print("Average loss at step {step} for last 250 steps: {loss}".format(step=step, loss=training_loss / 250))
            training_losses.append(training_loss / 250)
            training_loss = 0

plt.plot(training_losses)

# Close the Session when we're done.
sess.close()
