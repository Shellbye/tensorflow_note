# -*- coding:utf-8 -*-
# Created by shellbye on 2017/8/16.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import random

import tensorflow as tf

from PIL import Image
import numpy as np


class CaptchaData(object):

    def __init__(self, testing_set, training_set):
        self.testing_set = testing_set
        self.training_set = training_set
        self.training_images = self._read_images(self.training_set)
        self.testing_images = self._read_images(self.testing_set)

    def _read_images(self, image_list_file_name):
        train_imgs = []
        with codecs.open(image_list_file_name,
                         "r", encoding='utf8') as train_file:
            for line in train_file:
                image_path, label = line.split(" ")
                image_arry = load_image(image_path)
                train_imgs.append({
                    'image_arry': image_arry,
                    'label': label
                })
        return train_imgs

    def train_next_batch(self, count):
        batch_image, batch_label = [], []
        while len(batch_label) < count:
            c = random.choice(self.training_images)
            batch_image.append(c['image_arry'])
            batch_label.append(label_to_list(c['label']))
        return np.asanyarray(batch_image), np.asarray(batch_label)

    def test_images(self):
        t_imgs = []
        for img in self.testing_images:
            t_imgs.append(img['image_arry'])
        return np.asarray(t_imgs)

    def test_labels(self):
        t_imgs = []
        for img in self.testing_images:
            t_imgs.append(label_to_list(img['label']))
        return np.asarray(t_imgs)


def load_image(in_filename):
    img = Image.open(in_filename).convert("L")
    img.load()
    data = np.asarray(img, dtype="int32")
    data = data.flatten()
    return data

def label_to_list(label):
    l = [0 for i in range(10)]
    l[int(label)] = 1
    return l

def get_captcha():
    captcha = CaptchaData("../utils/testing_set.txt",
                          "../utils/training_set.txt")
    return captcha


def main(_):
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
    captcha = get_captcha()

    for _ in range(1000):
        batch_xs, batch_ys = captcha.train_next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    saver = tf.train.Saver()
    save_path = saver.save(sess, "models_diy/model1.ckpt")
    print("Model saved in file: ", save_path)
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    fd = {x: captcha.test_images(), y_: captcha.test_labels()}
    print(sess.run(accuracy, feed_dict=fd))

def real_test():
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    prediction = tf.argmax(y, 1)


    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saver.restore(sess, "models_diy/model1.ckpt")
        right, wrong = 0, 0
        with codecs.open("../utils/testing_set.txt", 'r', encoding='utf8') as f:
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
    # tf.app.run(main=main)
    real_test()
