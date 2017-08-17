# -*- coding:utf-8 -*-
# Created by shellbye on 2017/8/16.

import urllib
import urllib2
import os
import re
import argparse
import time
import sqlite3

server_url = 'http://127.0.0.1:8000/t/'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_img(file_path, count):
    with open(file_path, 'w') as f:
        f.seek(0)
        for i in range(count):
            print "start {}".format(i)
            response = urllib2.urlopen(server_url).read()

            src = re.findall("src=\"(.*?)\"", response)[0]
            hash_key = re.findall("value=\"(.*?)\"", response)[0]
            if not os.path.isdir(BASE_DIR + "/static/images"):
                os.mkdir(BASE_DIR + "/static/images")
            img_name = BASE_DIR + "/static/images/{}.jpg".format(time.time())

            urllib.urlretrieve("http://127.0.0.1:8000{}".format(src), img_name)

            f.write("{} {}\n".format(img_name, hash_key))
            # print num_label


def get_label(file_path):
    conn = sqlite3.connect('/Users/shellbye/Projects/tensorflow_note/db.sqlite3')
    cursor = conn.cursor()
    new_lines = []
    with open(file_path, 'r') as f:
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

    with open(file_path, "w") as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    # training set
    # get_img("training_set.txt", 2000)
    # get_label("training_set.txt")

    # test set
    get_img("testing_set.txt", 50)
    get_label("testing_set.txt")
