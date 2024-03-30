


import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Load the pre-trained model
model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

# Define transformations to preprocess the input image
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize the input image to the expected size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the output folder
output_folder = 'output_masks'
os.makedirs(output_folder, exist_ok=True)

# Load and preprocess the input image
image_path = r"D:\vtryon_workout\M3D-VTON\example\image\back_test1-removebg-preview.png"
image = Image.open(image_path)
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# Convert predictions to PIL image
output_image = transforms.ToPILImage()(output_predictions.byte())

# Save the output image to the output folder

output_image.save(r"D:\vtryon_workout\M3D-VTON\example\image-parse\back_test1-removebg-preview_label.png")




