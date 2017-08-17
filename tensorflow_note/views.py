# -*- coding:utf-8 -*-
# Created by shellbye on 2017/8/16.
from django.shortcuts import render_to_response

from tensorflow_note.forms import CaptchaTestForm


def some_view(request):
    if request.POST:
        form = CaptchaTestForm(request.POST)

        # Validate the form: the captcha field will automatically
        # check the input
        if form.is_valid():
            human = True
    else:
        form = CaptchaTestForm()

    return render_to_response('template.html', {"form": form})
