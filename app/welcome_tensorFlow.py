# -*- coding: utf-8 -*-
# @Time    : 2018/1/20 12:19
# @Author  : Ayan
# @Email   : hbally
# @File    : welcome_tensorFlow.py
# @Software: PyCharm

from __future__ import print_function
import tensorflow as tf
import os

# The default path for saving event files is the same folder of this python file.
tf.app.flags.DEFINE_string(
    'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
    'Directory where event logs are written to.')

# Store all elements in FLAG structure!
FLAGS = tf.app.flags.FLAGS
# The user is prompted to input an absolute path.
# os.path.expanduser is leveraged to transform '~' sign to the corresponding path indicator.
#       Example: '~/logs' equals to '/home/username/logs'
if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
    raise ValueError('You must assign absolute path for --log_dir')

# Defining some sentence!
welcome = tf.constant('Welcome to TensorFlow world!')

# Defining some constant values
a = tf.constant(5.0, name="a")
b = tf.constant(10.0, name="b")

# Some basic operations
x = tf.add(a, b, name="add")
y = tf.div(a, b, name="divide")

# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print("output: ", sess.run(welcome))

# Closing the writer.
writer.close()
sess.close()
