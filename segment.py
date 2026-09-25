"""Person segmentation map with torchvision's pretrained DeepLabV3-ResNet101.

usage: python segment.py example/image/me=person_whole_front.png --dataroot example
"""
import argparse
import os

import torch
import torchvision.transforms as transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('image', help='person image (.png)')
parser.add_argument('--dataroot', default='example')
opt = parser.parse_args()

model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open(opt.image).convert('RGB')
with torch.no_grad():
    output = model(preprocess(image).unsqueeze(0))['out'][0]

# Back to the input size (320x512) so the label map lines up with the photo
labels = transforms.ToPILImage()(output.argmax(0).byte()).resize(image.size, Image.NEAREST)

out_dir = os.path.join(opt.dataroot, 'image-parse')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, os.path.basename(opt.image).replace('.png', '_label.png'))
labels.save(out)
print('Saved', out)
