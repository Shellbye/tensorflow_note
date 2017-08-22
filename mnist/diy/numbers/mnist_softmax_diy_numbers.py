# -*- coding:utf-8 -*-
# Created by shellbye on 2017/8/16.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs

import tensorflow as tf

from mnist.diy.captcha import get_captcha
from mnist.diy.utils import load_image


def train(model_save_path):
    x = tf.placeholder(tf.float32, [None, 784])  # Here None means that a dimension can be of any length, which is row
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    captcha = get_captcha("/Users/shellbye/Projects/tensorflow_note/data/mnist/diy/numbers/")

    for _ in range(1000):
        batch_xs, batch_ys = captcha.train_next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    saver = tf.train.Saver()
    save_path = saver.save(sess, model_save_path)
    print("Model saved in file: ", save_path)
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    fd = {x: captcha.test_images(), y_: captcha.test_labels()}
    print(sess.run(accuracy, feed_dict=fd))

def real_test(model_save_path, testing_data):
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    prediction = tf.argmax(y, 1)


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess, model_save_path)
        right, wrong = 0, 0
        with codecs.open(testing_data, 'r', encoding='utf8') as f:
            for line in f:
                image_path, label = line.split(" ")
                img_data = load_image(image_path)
                # pre = prediction.eval(feed_dict={x: [img_data]}, session=sess)[0]
                pre = sess.run(prediction, feed_dict={x : [img_data]})[0]
                if pre == int(label):
                    right += 1
                else:
                    wrong += 1
                    print("predict {} {}, real {}".format(image_path, pre, label))
    print("right:{}, wrong:{}".format(right, wrong))

if __name__ == '__main__':
    # train("models_numbers/model.ckpt")
    real_test("models_numbers/model.ckpt",
              "/Users/shellbye/Projects/tensorflow_note/data/mnist/diy/numbers/testing_set.txt")
