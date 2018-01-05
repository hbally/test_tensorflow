# -*- coding: utf-8 -*-
# @Time    : 12/20/17 8:02 PM
# @Author  : liyao
# @Email   : hbally@126.com
# @File    : hello_tensorflow.py
# @Software: PyCharm

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hello = tf.constant('Hello,TensorFlow!')
sess = tf.Session()
print(sess.run(hello))