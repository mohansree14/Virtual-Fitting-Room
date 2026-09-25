"""Prepare a person photo for M3D-VTON: resize, palm mask, Sobel gradients, pose, placeholder depth.

usage: python human.py example/image/me=person_whole_front.png --dataroot example
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

from pose import add_model_args, save_pose


def out_path(dataroot, folder, name):
    os.makedirs(os.path.join(dataroot, folder), exist_ok=True)
    return os.path.join(dataroot, folder, name)


def resize_img(path):
    Image.open(path).resize((320, 512), Image.BICUBIC).convert('RGB').save(path)


def palm_mask(img_file, dataroot):
    # Largest skin-coloured region in HSV, assumed to be the hand
    hsv = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2HSV)
    mask_skin = cv2.inRange(hsv, np.array([0, 48, 80], np.uint8), np.array([20, 255, 255], np.uint8))
    contours, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print('No palm detected.')
        return
    mask = np.zeros_like(mask_skin)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    name = os.path.basename(img_file).replace('whole_front.png', 'palm_mask.png')
    cv2.imwrite(out_path(dataroot, 'palm-mask', name), mask)


def sobel(img_file, dataroot):
    gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    name = os.path.basename(img_file)
    for axis, (dx, dy) in (('x', (1, 0)), ('y', (0, 1))):
        grad = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=3)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(out_path(dataroot, 'image-sobel', name.replace('.png', f'_sobel{axis}.png')), grad)


def placeholder_depth(img_file, dataroot):
    # ponytail: grayscale stand-in, not a real depth estimate; swap in a monocular depth model for real results
    gray = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    np.save(out_path(dataroot, 'depth', os.path.basename(img_file).replace('.png', '_depth.npy')), gray)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='person image (.png), resized in place to 320x512')
    parser.add_argument('--dataroot', default='example')
    add_model_args(parser)
    opt = parser.parse_args()

    resize_img(opt.image)
    palm_mask(opt.image, opt.dataroot)
    sobel(opt.image, opt.dataroot)
    save_pose(opt.image, opt.dataroot, opt.prototxt, opt.caffemodel)
    placeholder_depth(opt.image, opt.dataroot)
    print('Done:', opt.image)
