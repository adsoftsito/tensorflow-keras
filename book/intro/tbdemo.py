from __future__ import print_function
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

a = tf.constant(10, name='a')
b = tf.constant(90, name='b')
y = tf.Variable(a+b*2, name='y')

model = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as session:
    merged = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter("/tmp/tensorflowlogs", session.graph)
    session.run(model)
    print(session.run(y))
