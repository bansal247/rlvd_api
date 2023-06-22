#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: sudhanshukumar
"""

import cv2
import base64
import torch

import get_model
import matplotlib.pyplot as plt
from flask import jsonify

from lr_results import lr


class rlvd:
    def __init__(self, filename):
        self.filename = filename
        self.lr_predict = lr()

    def get_number(self, results, img):
        min, ymax, ymin = None, None, None
        for i in range(len(results.xywhn[0])):
            # print(results.xywhn[0][i][5],results.xywhn[0][i][1],results.xywhn[0][i][1]+results.xywhn[0][i][3])
            if int(results.xywhn[0][i][5]) == 1:
                ymin = results.xywhn[0][i][1]
                ymax = results.xywhn[0][i][1] + results.xywhn[0][i][3]
            # print(ymin,ymax)
        X, Y, W, H = None, None, None, None
        d = None
        for i in range(len(results.xywhn[0])):
            if int(results.xywhn[0][i][5]) == 2 and results.xywhn[0][i][1] + results.xywhn[0][i][3] > ymin and results.xywhn[0][i][1] + results.xywhn[0][i][3] < ymax:
                d = i

        if d:
            cropped_image = img[int(results.xyxyn[0][d][1] * 720):int(results.xyxyn[0][d][3] * 720),
                            int(results.xyxyn[0][d][0] * 1280):int(results.xyxyn[0][d][2] * 1280)]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            plt.imsave('cropped_image.jpg', cropped_image)
            #reader = easyocr.Reader(['en'])
            try:
                number = self.lr_predict.get_result('cropped_image.jpg')
            except:
                number = '-'
        else:
            number = '-'
        return number

    def get_results(self):
        # load model
        model = get_model.model

        imagename = self.filename
        test_image = cv2.imread(imagename)
        with torch.inference_mode():
            results = model(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))

            results.save('result_image')

        number = self.get_number(results, test_image)
        with open("result_image//image0.jpg", "rb") as img_file:
            #print(img_file.read())
            my_string = base64.b64encode(img_file.read())
        json_results = {
            "image": my_string.decode(),
            "number": number
        }
        json_results = jsonify(json_results)
        return json_results
