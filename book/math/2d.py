from __future__ import print_function
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

tensor_2d = np.array([(1, 2, 3, 4), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14,15)])

print (tensor_2d)
print (tensor_2d[3][3])
#print (tensor_2d.ndim)
#print (tensor_2d.shape)
print (tensor_2d[0:1, 0:4])

#tf_tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)

#with tf.compat.v1.Session() as sess:
#    print (sess.run(tf_tensor));
#    print (sess.run(tf_tensor[0]));
#    print (sess.run(tf_tensor[2]));
