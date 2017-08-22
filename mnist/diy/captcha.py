# -*- coding:utf-8 -*-
# Created by shellbye on 2017/8/18.
import codecs
import random

import numpy as np

from mnist.diy.utils import label_to_list, load_image, label_to_list_alphabet


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

    def train_next_batch(self, count, alpha=False):
        batch_image, batch_label = [], []
        while len(batch_label) < count:
            c = random.choice(self.training_images)
            batch_image.append(c['image_arry'])
            if alpha:
                batch_label.append(label_to_list_alphabet(c['label']))
            else:
                batch_label.append(label_to_list(c['label']))
        return np.asanyarray(batch_image), np.asarray(batch_label)

    def test_images(self):
        t_imgs = []
        for img in self.testing_images:
            t_imgs.append(img['image_arry'])
        return np.asarray(t_imgs)

    def test_labels(self, alpha=False):
        t_imgs = []
        for img in self.testing_images:
            if alpha:
                t_imgs.append(label_to_list_alphabet(img['label']))
            else:
                t_imgs.append(label_to_list(img['label']))
        return np.asarray(t_imgs)


def get_captcha(data_path):
    captcha = CaptchaData(data_path + "testing_set.txt",
                          data_path + "training_set.txt")
    return captcha

if __name__ == '__main__':
    pass
