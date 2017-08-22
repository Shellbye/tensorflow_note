# -*- coding:utf-8 -*-
# Created by shellbye on 2017/8/22.
from utils.generate_pic import get_label, get_img

if __name__ == '__main__':
    # training set
    get_img("data/mnist/diy/numbers/", "training_set.txt", 100, "images")
    get_label("data/mnist/diy/numbers/", "training_set.txt")

    # test set
    get_img("data/mnist/diy/numbers/", "testing_set.txt", 10, "images")
    get_label("data/mnist/diy/numbers/", "testing_set.txt")
