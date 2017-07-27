import numpy as np
import tensorflow as tf

# this works and learns somtnihg but I am not sure if everythig is correct.
# Need to check output, input and apply to my own data. 


# Unit test for Phased LSTM
flags = tf.flags
flags.DEFINE_integer('batch_size', 32, 'batch size')  # 32
flags.DEFINE_float('max_length', 125, 'max length of sin waves')
flags.DEFINE_float('min_length', 50, 'min length of sine waves')
flags.DEFINE_float('max_f_off', 100, 'max frequency for the off set')
flags.DEFINE_float('min_f_off', 1, 'min frequency for the off set')
flags.DEFINE_float('max_f_on', 5, 'max frequency for the on set')
flags.DEFINE_float('min_f_on', 6, 'min frequency for the on set')
FLAGS = flags.FLAGS

# Net Params
n_input = 1
n_out = 2
n_hidden = 16  # hidden units in the recurrent layer
n_epochs = 30
b_per_epoch = 80


def gen_async_sin(batch_size=32, on_target_T=(5, 6), off_target_T=(1, 100), max_len=125, min_len=85):
    half_batch = int(batch_size / 2)
    full_length = off_target_T[1] - on_target_T[1] + on_target_T[0] - off_target_T[0]
    # generate random periods
    posTs = np.random.uniform(on_target_T[0], on_target_T[1], half_batch)
    size_low = np.floor((on_target_T[0] - off_target_T[0]) * half_batch / full_length).astype('int32')
    size_high = np.ceil((off_target_T[1] - on_target_T[1]) * half_batch / full_length).astype('int32')
    low_vec = np.random.uniform(off_target_T[0], on_target_T[0], size_low)
    high_vec = np.random.uniform(on_target_T[1], off_target_T[1], size_high)
    negTs = np.hstack([low_vec,
                       high_vec])
    # generate random lengths
    lens = np.random.uniform(min_len, max_len, batch_size)

    # generate random number of samples
    samples = np.random.uniform(min_len, max_len, batch_size).astype('int32')

    start_times = np.array([np.random.uniform(0, max_len - duration) for duration in lens])
    x = np.zeros((batch_size, max_len, 1))
    y = np.zeros((batch_size, 2))
    t = np.zeros((batch_size, max_len, 1))
    for i, s, l, n in zip(range(batch_size), start_times, lens, samples):
        time_points = np.reshape(np.sort(np.random.uniform(s, s + l, n)), [-1, 1])

        if i < half_batch:  # positive
            _tmp_x = np.squeeze(np.sin(time_points * 2 * np.pi / posTs[i]))
            x[i, :len(_tmp_x), 0] = _tmp_x
            t[i, :len(_tmp_x), 0] = np.squeeze(time_points)
            y[i, 0] = 1.
        else:
            _tmp_x = np.squeeze(np.sin(time_points * 2 * np.pi / negTs[i - half_batch]))
            x[i, :len(_tmp_x), 0] = _tmp_x
            t[i, :len(_tmp_x), 0] = np.squeeze(time_points)
            y[i, 1] = 1.

    return x, t, y, samples, posTs, negTs


# inputs
x = tf.placeholder(tf.float32, shape=(None, None, 1))
t = tf.placeholder(tf.float32, shape=(None, None, 1))

# # length of the samples -> for dynamic_rnn
lens = tf.placeholder(tf.int32, [None])

# labels
y = tf.placeholder(tf.float32, [None, 2])

# Let's define the training and testing operations
cell = tf.contrib.rnn.PhasedLSTMCell(num_units=n_hidden, use_peepholes=True)
# outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=(x, t), dtype=tf.float32, sequence_length=lens)
outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=(x, t), dtype=tf.float32)

t_output = tf.transpose(a=outputs, perm=[1, 0, 2])
predictions = tf.layers.dense(inputs=t_output[-1], units=n_out)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

# evaluation
correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# run the model
sess = tf.InteractiveSession()
sess.run(fetches=tf.global_variables_initializer())

# training loop
for step in range(n_epochs):
    train_cost = 0
    train_acc = 0
    for i in range(b_per_epoch):
        batch_xs, batch_ts, batch_ys, _, _, _ = gen_async_sin(FLAGS.batch_size, [FLAGS.min_f_on, FLAGS.max_f_on],
                                                                 [FLAGS.min_f_off, FLAGS.max_f_off],
                                                                 FLAGS.max_length,
                                                                 FLAGS.min_length)

        res = sess.run([optimizer, cross_entropy, accuracy],
                       feed_dict={x: batch_xs,
                                  t: batch_ts,
                                  y: batch_ys})
        train_cost += res[1] / b_per_epoch
        train_acc += res[2] / b_per_epoch

        # test accuracy
        test_xs, test_ts, test_ys, _, _, _ = gen_async_sin(FLAGS.batch_size * 10, [FLAGS.min_f_on, FLAGS.max_f_on],
                                                                [FLAGS.min_f_off, FLAGS.max_f_off],
                                                                FLAGS.max_length, FLAGS.min_length)

        loss_test = cross_entropy.eval(feed_dict={x: test_xs, t: test_ts, y: test_ys})
        acc_test = accuracy.eval(feed_dict={x: test_xs, t: test_ts, y: test_ys})

        msg = "step: {e}/{steps}, train loss: {tr_l}, train accuracy: {tr_a} test loss: {te_l}, test accuracy: {te_a}".format(
            e=step, steps=n_epochs, tr_l=train_cost, tr_a=train_acc, te_l=loss_test, te_a=acc_test)
        print(msg)
