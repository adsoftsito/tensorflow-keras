from __future__ import print_function
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np

matrix1 = np.array([(2,2,2),(2,2,2),(2,2,2)])
matrix2 = np.array([(1,1,1),(1,1,1),(1,1,1)])
matrix3 = np.array([(2,7,2),(1,4,2),(9,0,2)], dtype='float32')

print ('matrix1 ')
print (matrix1)

print ('matrix2 ')
print (matrix2)

matrix1 = tf.constant(matrix1)
matrix2 = tf.constant(matrix2)

matrix_product = tf.matmul(matrix1, matrix2)
matrix_sum = tf.add(matrix1, matrix2)
matrix_det = tf.matrix_determinant(matrix3)

with tf.compat.v1.Session() as sess:
    result1 = sess.run(matrix_product)
    result2 = sess.run(matrix_sum)
    result3 = sess.run(matrix_det)
    print('matmul')
    print (result1)
    print('add')
    print(result2)
    print('det')
    print(result3)


