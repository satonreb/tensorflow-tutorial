import tensorflow as tf

a = tf.placeholder(dtype=tf.float32, shape=None, name="a")
b = tf.placeholder(dtype=tf.float32, shape=None, name="b")
c = tf.add(a, b, name="r")


av = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="av")
bv = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="bv")
cv = tf.add(x=av, y=bv, name="cv")


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

feed = {a: 2, b: 9}
print(sess.run(fetches=c, feed_dict=feed))

feedv = {av: [[1], [2], [3], [4]], bv: [[5], [6], [7], [8]]}
print(sess.run(fetches=cv, feed_dict=feedv))
