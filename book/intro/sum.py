from __future__ import print_function
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x = 1;
y = x + 9;
print(y);

x = tf.constant(1, name='x')
y = tf.Variable(x+1, name='y')

model = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as session:
    session.run(model)
    print(session.run(x))
    print(session.run(y))
