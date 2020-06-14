from __future__ import print_function
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x = tf.compat.v1.placeholder(tf.float32)
y = 3 * x * x

var_grad = tf.gradients(y, x);

with tf.compat.v1.Session() as sess:
    var_grad_val = sess.run(var_grad, feed_dict={x:4})
    print (var_grad_val);
    print (var_grad);
    #print (sess.run(tf_tensor[2]));
