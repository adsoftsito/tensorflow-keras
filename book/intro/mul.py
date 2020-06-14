from __future__ import print_function
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

a = tf.compat.v1.placeholder("int32");
b = tf.compat.v1.placeholder("int32");

y = tf.multiply(a, b)

session = tf.compat.v1.Session()
print(session.run(y, feed_dict = {a:0, b:5}))
