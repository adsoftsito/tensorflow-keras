from __future__ import print_function
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

tensor_1d = np.array([1.3, 1, 4.0, 23.09, 2.7, 3.14])

print (tensor_1d)
print (tensor_1d[0])
print (tensor_1d.ndim)
print (tensor_1d.shape)
print (tensor_1d.dtype)

tf_tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)

with tf.compat.v1.Session() as sess:
    print (sess.run(tf_tensor));
    print (sess.run(tf_tensor[0]));
    print (sess.run(tf_tensor[2]));
