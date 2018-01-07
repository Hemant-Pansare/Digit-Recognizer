# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 17:32:39 2018

@author: Hemant
"""

# Importing required packages 
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# Reading data into python
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


# Creating placeholders for input data and variables for parameters
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder(tf.float32, [None, 10])


# Calculating probabilities for outputs
y = tf.nn.softmax(tf.matmul(x, W) + b)


# Defining the Cost function and Optimization technique
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = (tf.matmul(x, W) + b), labels = y_))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)


# Starting TF session
sess = tf.InteractiveSession()


# Initializing variables
sess.run(tf.global_variables_initializer())


# Running loop to calculate weights and bias
for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, {x : batch_xs, y_ : batch_ys})


# Checking accuracy of test data
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict = {x : mnist.test.images, y_ : mnist.test.labels}))

curr_W, curr_b = sess.run([W, b])
print("W : %s    b : %s" %(curr_W, curr_b))