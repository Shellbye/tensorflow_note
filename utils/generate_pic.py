# -*- coding:utf-8 -*-
# Created by shellbye on 2017/8/16.

import urllib
import urllib2
import os
import re
import time
import sqlite3

server_url = 'http://127.0.0.1:8000/t/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_img(file_path, label_file, count, image_file_path):
    top_dir = "{}/{}".format(BASE_DIR, file_path)
    if not os.path.isdir(top_dir):
        os.makedirs(top_dir)
    image_dir = "{}{}".format(top_dir, image_file_path)
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    with open("{}{}".format(top_dir, label_file), 'w+') as f:
        f.seek(0)
        for i in range(count):
            print("start {}".format(i))
            response = urllib2.urlopen(server_url).read()
            src = re.findall("src=\"(.*?)\"", response)[0]
            hash_key = re.findall("value=\"(.*?)\"", response)[0]
            img_name = "{}/{}.jpg".format(image_dir, time.time())
            urllib.urlretrieve("http://127.0.0.1:8000{}".format(src), img_name)
            f.write("{} {}\n".format(img_name, hash_key))


def get_label(file_path, label_file):
    top_dir = "{}/{}".format(BASE_DIR, file_path)
    if not os.path.isdir(top_dir):
        os.makedirs(top_dir)
    conn = sqlite3.connect('/Users/shellbye/Projects/tensorflow_note/db.sqlite3')
    cursor = conn.cursor()
    new_lines = []
    f_label = "{}{}".format(top_dir, label_file)
    with open(f_label, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            img_path, hash_key = line.strip().split(" ")
            cursor.execute(
                "select challenge from captcha_captchastore where hashkey='{}';".format(hash_key.strip()))
            row = cursor.fetchone()
            if row:
                num_label = row[0]
                line = "{} {}\n".format(img_path, num_label)
                new_lines.append(line)
            else:
                print hash_key

    with open(f_label, "w") as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    pass
    # # training set
    # get_img("data/mnist/diy/alphabet/", "training_set.txt", 100, "images")
    # get_label("data/mnist/diy/alphabet/", "training_set.txt")
    #
    # # test set
    # get_img("data/mnist/diy/alphabet/", "testing_set.txt", 10, "images")
    # get_label("data/mnist/diy/alphabet/", "testing_set.txt")
