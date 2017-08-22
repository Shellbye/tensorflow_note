# -*- coding:utf-8 -*-
# Created by shellbye on 2017/8/18.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

import tensorflow as tf

from mnist.diy.captcha import get_captcha
from mnist.diy.utils import load_image


def train(model_save_path):
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 26]))
    b = tf.Variable(tf.zeros([26]))
    y = tf.matmul(x, W) + b

    y_real = tf.placeholder(tf.float32, [None, 26])

    # https://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
    loss_function = tf.nn.softmax_cross_entropy_with_logits(labels=y_real, logits=y)
    cross_entropy = tf.reduce_mean(loss_function)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    captcha = get_captcha("/Users/shellbye/Projects/tensorflow_note/data/mnist/diy/alphabet/")
    for _ in range(1000):
        batch_xs, batch_ys = captcha.train_next_batch(100, alpha=True)
        sess.run(train_step, feed_dict={x: batch_xs, y_real: batch_ys})
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_save_path)
    print("Model saved in file: ", save_path)
    # Test trained model
    # tf.argmax(y, 1) the max value's index
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_real, 1))  # element-wise compare
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    fd = {x: captcha.test_images(), y_real: captcha.test_labels(alpha=True)}
    print(sess.run(accuracy, feed_dict=fd))

def real_test(model_save_path, testing_data):
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 26]))
    b = tf.Variable(tf.zeros([26]))
    y = tf.matmul(x, W) + b

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess, model_save_path)
        right, wrong = 0, 0
        with open(testing_data, "r") as f:
            for line in f:
                line = line.replace("\n", "")
                image, label = line.split(" ")
                image_data = load_image(image)
                pre = sess.run(tf.arg_max(y, 1), feed_dict={x: [image_data]})[0]
                if pre == string.ascii_uppercase.index(label):
                    right += 1
                else:
                    wrong += 1
                    print("predict {} {}, real {}".format(image, pre, label))
    print("right:{}, wrong:{}".format(right, wrong))

if __name__ == '__main__':
    # train("models_alphabet/model.ckpt")
    real_test("models_alphabet/model.ckpt",
              "/Users/shellbye/Projects/tensorflow_note/data/mnist/diy/alphabet/testing_set.txt")
