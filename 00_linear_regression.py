import tensorflow as tf
import numpy as np

# Script shows how to use TensorFlow for linear regression  for one feature and one class

# CUSTOMIZABLE: Collect/Prepare data
steps = 1000
actual_W = 2
actual_b = 10
learning_rate = 0.000001

# Model construction
# placeholder for input variable
x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')

# definition of weight matrix and bias vector
W = tf.Variable(initial_value=tf.zeros([1, 1]), name='weight')
b = tf.Variable(initial_value=tf.zeros([1]), name='bias')

# definition of outcome variable ( y = W * x + b )
y_pred = tf.add(x=tf.matmul(a=x, b=W), y=b)

# placeholder for true outcome values
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_true')

# definition for cost function (loss function ...) sum((yt -yp)^2)
cost = tf.reduce_mean(input_tensor=tf.square(x=(y_true-y_pred)), name='cost_func')

# definition for gradient decent object with step == learning_rate and cost as minimization target
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=cost)

# for performance reasons and other things Tensorflow has to be executed within session
sess = tf.Session()

# define variable initialisation
init = tf.global_variables_initializer()

# perform actual variable initialisation
sess.run(fetches=init)

# training steps
for i in range(steps):
    # Create fake data for y = W.x + b where W = actual_W, b = actual_b
    xs = np.array([[i]])
    ys = np.array([[actual_W * i + actual_b]])

    # training
    feed = {x: xs, y_true: ys}
    sess.run(fetches=train_step, feed_dict=feed)

    print("Iteration {iter} and cost is {cost} ".format(iter=i, cost=sess.run(fetches=cost, feed_dict=feed)))

print("Predicted: W={W} and  b={b}".format(W=sess.run(fetches=W), b=sess.run(fetches=b)))
