"""Resize garment images to 320x512 and build their masks by thresholding.

usage: python cloth.py example/cloth/shirt.jpg [more.jpg ...] --dataroot example
"""
import argparse
import os

import cv2
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('images', nargs='+', help='garment images (.jpg) on a light background, resized in place')
parser.add_argument('--dataroot', default='example')
parser.add_argument('--threshold', type=int, default=200, help='pixels brighter than this count as background')
opt = parser.parse_args()

mask_dir = os.path.join(opt.dataroot, 'cloth-mask')
os.makedirs(mask_dir, exist_ok=True)

for path in opt.images:
    Image.open(path).resize((320, 512), Image.BICUBIC).convert('RGB').save(path)
    gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, opt.threshold, 255, cv2.THRESH_BINARY_INV)
    out = os.path.join(mask_dir, os.path.basename(path).replace('.jpg', '_mask.jpg'))
    cv2.imwrite(out, mask)
    print('Saved', out)
