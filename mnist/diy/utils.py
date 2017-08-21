# -*- coding:utf-8 -*-
# Created by shellbye on 2017/8/18.
import string

from PIL import Image
import numpy as np


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


def label_to_list_alphabet(label):
    l = [0 for i in range(26)]
    label = label.strip()
    l[string.ascii_uppercase.index(label)] = 1
    return l

if __name__ == '__main__':
    pass
