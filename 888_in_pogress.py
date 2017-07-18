"""
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
Inspired by https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
# noinspection PyUnresolvedReferences
import seaborn as sns

import tensorflow as tf


def generate_sample(f: Optional[float] = 1.0, t0: Optional[float] = None, batch_size: int = 1,
                    predict: int = 50, samples: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates data samples.
    :param f: The frequency to use for all time series or None to randomize.
    :param t0: The time offset to use for all time series or None to randomize.
    :param batch_size: The number of time series to generate.
    :param predict: The number of future samples to generate.
    :param samples: The number of past (and current) samples to generate.
    :return: Tuple that contains the past times and values as well as the future times and values. In all outputs,
             each row represents one time series of the batch.
    """
    Fs = 100

    T = np.empty((batch_size, samples))
    Y = np.empty((batch_size, samples))
    FT = np.empty((batch_size, predict))
    FY = np.empty((batch_size, predict))

    _t0 = t0
    for i in range(batch_size):
        t = np.arange(0, samples + predict) / Fs
        if _t0 is None:
            t0 = np.random.rand() * 2 * np.pi
        else:
            t0 = _t0 + i/float(batch_size)

        freq = f
        if freq is None:
            freq = np.random.rand() * 3.5 + 0.5

        y = np.sin(2 * np.pi * freq * (t + t0))

        T[i, :] = t[0:samples]
        Y[i, :] = y[0:samples]

        FT[i, :] = t[samples:samples + predict]
        FY[i, :] = y[samples:samples + predict]

    return T, Y, FT, FY

# def RNN(x, weights, biases, n_input, n_steps, n_hidden):
#
#     # Prepare data shape to match `rnn` function requirements
#     # Current data input shape: (batch_size, n_steps, n_input)
#     # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
#
#     # Permuting batch_size and n_steps
#     x = tf.transpose(x, [1, 0, 2])
#     # Reshaping to (n_steps*batch_size, n_input)
#     x = tf.reshape(x, [-1, n_input])
#     # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
#     x = tf.split(x, n_steps, axis=0)
#
#     # Define a lstm cell with tensorflow
#     lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
#
#     # Get lstm cell output
#     outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
#
#     # Linear activation, using rnn inner loop last output
#     return tf.nn.bias_add(tf.matmul(outputs[-1], weights['out']), biases['out'])



# Parameters
learning_rate = 0.001
training_iters = 20000
batch_size = 50
display_step = 100

# Network Parameters
n_input = 1  # input is sin(x)
n_steps = 100  # timesteps
n_hidden = 100  # hidden layer num of features
n_outputs = 50  # output is sin(x+1)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_outputs])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_outputs]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_outputs]))
}

# Prepare data shape to match `rnn` function requirements
# Current data input shape: (batch_size, n_steps, n_input)
# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

# Permuting batch_size and n_steps
lstm_x = tf.transpose(x, [1, 0, 2])
# Reshaping to (n_steps*batch_size, n_input)
lstm_x = tf.reshape(lstm_x, [-1, n_input])
# Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
lstm_x = tf.split(lstm_x, n_steps, axis=0)

# last_x_2 = tf.unstack(value=x, axis=1)

# Define a lstm cell with tensorflow
lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

# Get lstm cell output
outputs, states = rnn.static_rnn(lstm_cell, lstm_x, dtype=tf.float32)

# Linear activation, using rnn inner loop last output
pred = tf.nn.bias_add(tf.matmul(outputs[-1], weights['out']), biases['out'])

# pred = RNN(x, weights, biases, n_input, n_steps, n_hidden)

# Define loss (Euclidean distance) and optimizer
individual_losses = tf.reduce_sum(tf.squared_difference(pred, y), reduction_indices=1)
loss = tf.reduce_mean(individual_losses)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        _, batch_x, __, batch_y = generate_sample(f=1.0, t0=None, batch_size=batch_size, samples=n_steps, predict=n_outputs)

        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        batch_y = batch_y.reshape((batch_size, n_outputs))

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss
            loss_value = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss_value))
        step += 1
    print("Optimization Finished!")

    # Test the prediction
    n_tests = 3
    for i in range(1, n_tests+1):
        plt.subplot(n_tests, 1, i)
        t, y, next_t, expected_y = generate_sample(f=i, t0=None, samples=n_steps, predict=n_outputs)

        test_input = y.reshape((1, n_steps, n_input))
        prediction = sess.run(pred, feed_dict={x: test_input})

        # remove the batch size dimensions
        t = t.squeeze()
        y = y.squeeze()
        next_t = next_t.squeeze()
        prediction = prediction.squeeze()

        plt.plot(t, y, color='black')
        plt.plot(np.append(t[-1], next_t), np.append(y[-1], expected_y), color='green', linestyle=':')
        plt.plot(np.append(t[-1], next_t), np.append(y[-1], prediction), color='red')
        plt.ylim([-1.1, 1.1])
        plt.xlabel('time [t]')
        plt.ylabel('signal')

    plt.show()
