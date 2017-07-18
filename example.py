import numpy as np
import tensorflow as tf


class SimpleDataIterator():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor + n - 1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor + n - 1]
        self.cursor += n
        return res['as_numbers'], res['gender'] * 3 + res['age_bracket'], res['length']


class PaddedDataIterator(SimpleDataIterator):
    def next_batch(self, n):
        if self.cursor + n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.ix[self.cursor:self.cursor + n - 1]
        self.cursor += n

        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]

        return x, res['gender'] * 3 + res['age_bracket'], res['length']


def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def train_graph(graph, batch_size=256, num_epochs=10, iterator=PaddedDataIterator):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tr = iterator(train)
        te = iterator(test)

        step, accuracy = 0, 0
        tr_losses, te_losses = [], []
        current_epoch = 0
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: 0.6}
            accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
            accuracy += accuracy_

            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                step, accuracy = 0, 0

                # eval test set
                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    batch = te.next_batch(batch_size)
                    feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2]}
                    accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                    accuracy += accuracy_

                te_losses.append(accuracy / step)
                step, accuracy = 0, 0
                print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])

    return tr_losses, te_losses


def build_seq2seq_graph(
        vocab_size=len(vocab),
        state_size=64,
        batch_size=256,
        num_classes=6):
    reset_graph()

    # Placeholders
    x = tf.placeholder(tf.int32, [batch_size, None])  # [batch_size, num_steps]
    seqlen = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.constant(1.0)

    # Tile the target indices
    # (in a regular seq2seq model, our targets placeholder might have this shape)
    y_y_ = tf.tile(tf.expand_dims(y, 1), [1, tf.shape(x)[1]])  # [batch_size, num_steps]

    """
    Create a mask that we will use for the cost function

    This mask is the same shape as x and y_, and is equal to 1 for all non-PAD time
    steps (where a prediction is made), and 0 for all PAD time steps (no pred -> no loss)
    The number 30, used when creating the lower_triangle_ones matrix, is the maximum
    sequence length in our dataset
    """
    lower_triangular_ones = tf.constant(np.tril(np.ones([30, 30])), dtype=tf.float32)
    seqlen_mask = tf.slice(tf.gather(lower_triangular_ones, seqlen - 1), [0, 0], [batch_size, tf.reduce_max(seqlen)])

    # Embedding layer
    embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)

    # RNN
    cell = tf.nn.rnn_cell.GRUCell(state_size)
    init_state = tf.get_variable('init_state', [1, state_size],
                                 initializer=tf.constant_initializer(0.0))
    init_state = tf.tile(init_state, [batch_size, 1])
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen,
                                                 initial_state=init_state)
    # reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y_, [-1])

    # Softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(rnn_outputs, W) + b

    preds = tf.nn.softmax(logits)

    # To calculate the number correct, we want to count padded steps as incorrect
    correct = tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), tf.int32), y_reshaped), tf.int32) * \
              tf.cast(tf.reshape(seqlen_mask, [-1]), tf.int32)

    # To calculate accuracy we want to divide by the number of non-padded time-steps,
    # rather than taking the mean
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(seqlen, tf.float32))

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped)
    loss = loss * tf.reshape(seqlen_mask, [-1])

    # To calculate average loss, we need to divide by number of non-padded time-steps,
    # rather than taking the mean
    loss = tf.reduce_sum(loss) / tf.reduce_sum(seqlen_mask)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }


g = build_seq2seq_graph()
tr_losses, te_losses = train_graph(g, iterator=BucketedDataIterator)
