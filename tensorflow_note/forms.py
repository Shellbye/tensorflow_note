# -*- coding:utf-8 -*-
# Created by shellbye on 2017/8/16.
from django import forms
from captcha.fields import CaptchaField

class CaptchaTestForm(forms.Form):
    captcha = CaptchaField()

if __name__ == '__main__':
    pass
