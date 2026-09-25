"""Extract 18 COCO body keypoints with OpenCV DNN and save them as OpenPose JSON.

usage: python pose.py example/image/me=person_whole_front.png --dataroot example
"""
import argparse
import json
import os

import cv2


def predict_keypoints(img_file, prototxt, caffemodel, threshold=0.05, size=368):
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    img = cv2.imread(img_file)
    img_h, img_w, _ = img.shape
    net.setInput(cv2.dnn.blobFromImage(img, 1.0 / 255, (size, size), (0, 0, 0), swapRB=False, crop=False))
    output = net.forward()
    H, W = output.shape[2], output.shape[3]

    points = []
    for idx in range(18):  # 18 keypoints in the COCO model
        _, prob, _, point = cv2.minMaxLoc(output[0, idx, :, :])  # peak of the confidence map
        if prob > threshold:
            points += [img_w * point[0] / W, img_h * point[1] / H, prob]
        else:
            points += [0, 0, 0]
    return points


def save_pose(img_file, dataroot, prototxt, caffemodel):
    out_dir = os.path.join(dataroot, 'pose')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, os.path.basename(img_file).replace('.png', '_keypoints.json'))
    points = predict_keypoints(img_file, prototxt, caffemodel)
    with open(out_file, 'w') as f:
        json.dump({"version": 1, "people": [{"pose_keypoints_2d": points}]}, f, indent=4)
    print('Saved', out_file)


def add_model_args(parser):
    parser.add_argument('--prototxt', default='openpose_pose_coco.prototxt')
    parser.add_argument('--caffemodel', default='pose_iter_440000.caffemodel')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='person image (.png)')
    parser.add_argument('--dataroot', default='example')
    add_model_args(parser)
    opt = parser.parse_args()
    save_pose(opt.image, opt.dataroot, opt.prototxt, opt.caffemodel)
