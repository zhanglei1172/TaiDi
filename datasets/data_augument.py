#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from config import *
IMG_WIDTH = 512
IMG_HEIGHT = 512

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((IMG_WIDTH / 2, IMG_HEIGHT/ 2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (IMG_WIDTH, IMG_HEIGHT))
    yb = cv2.warpAffine(yb, M_rotate, (IMG_WIDTH, IMG_HEIGHT))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3));
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, IMG_HEIGHT)
        temp_y = np.random.randint(0, IMG_WIDTH)
        img[temp_x][temp_y] = 1.
    return img



