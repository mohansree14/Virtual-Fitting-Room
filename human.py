import cv2
from PIL import Image
def resize_img(path):
    c = Image.open(path)
    c = c.resize((320, 512), Image.BICUBIC).convert('RGB')
    c.save(path)

path_front=r"D:\vtryon_workout\M3D-VTON\images\person\front_test1-removebg-preview.png"
resize_img(path_front)
path_back=r"D:\vtryon_workout\M3D-VTON\images\person\back_test1-removebg-preview.png"
resize_img(path_back)

#----------------------------------------------------------------------------------------------------------------
#palm mask

import cv2
import numpy as np

# Read the input image
image = cv2.imread(r"D:\vtryon_workout\M3D-VTON\images\person\front_test1-removebg-preview.png")
image_copy = image.copy()  # Create a copy for visualization

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for skin color in HSV
lower_skin = np.array([0, 48, 80], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Create a binary mask for skin regions
mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)

# Find contours in the skin mask
contours, _ = cv2.findContours(mask_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours to find the largest one (assumed to be the hand)
if len(contours) > 0:
    max_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    mask_palm = np.zeros_like(mask_skin)
    cv2.drawContours(mask_palm, [max_contour], -1, 255, thickness=cv2.FILLED)

    # Optional: Smooth the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask_palm = cv2.morphologyEx(mask_palm, cv2.MORPH_OPEN, kernel)

    # Save the palm mask as an image
    cv2.imwrite(r"D:\vtryon_workout\M3D-VTON\examples_3d\palm-mask\front_test1-removebg-preview.png", mask_palm)
    print("Palm mask saved as 'palm_mask.jpg'.")
else:
    print("No palm detected.")


#----------------------------------------------------------------------------------------------------------------------
#image sobel
import cv2
import numpy as np

# Read the input image
image = cv2.imread(r"D:\vtryon_workout\M3D-VTON\images\person\front_test1-removebg-preview.png", cv2.IMREAD_GRAYSCALE)

# Apply Sobel operator
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel operator along X-axis
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel operator along Y-axis

# Convert to the appropriate data type for saving
sobel_x = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
sobel_y = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Save Sobel X and Sobel Y images
cv2.imwrite(r"D:\vtryon_workout\M3D-VTON\examples_3d\image-sobel\front_test1-removebg-preview_x.png", sobel_x)
cv2.imwrite(r"D:\vtryon_workout\M3D-VTON\examples_3d\image-sobel\front_test1-removebg-preview_y.png", sobel_y)

print("Sobel X and Sobel Y images saved successfully.")

#image parse
#----------------------------------------------------------------------------------------------------------------------



#pose
#----------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import json
import cv2

class general_pose_model(object):
    def __init__(self, modelpath):
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.05
        self.pose_net = self.general_coco_model(modelpath)
        
    def general_coco_model (self, modelpath):
        prototxt = r"D:\vtryon_workout\M3D-VTON\examples_3d\openpose_pose_coco.prototxt"
        caffemodel = r"D:\vtryon_workout\M3D-VTON\examples_3d\pose_iter_440000.caffemodel"
        coco_model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        return coco_model

    def predict(self, imgfile):
        img_cv2 = cv2.imread(imgfile)
        img_height, img_width, _ = img_cv2.shape
        inpBlob = cv2.dnn.blobFromImage(img_cv2, 
                                        1.0 / 255, 
                                        (self.inWidth, self.inHeight),
                                        (0, 0, 0), 
                                        swapRB=False, 
                                        crop=False)
        self.pose_net.setInput(inpBlob)
        self.pose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.pose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

        output = self.pose_net.forward()

        H = output.shape[2]
        W = output.shape[3]
        
        points = []
        for idx in range(18):  # 18 keypoints in COCO model
            probMap = output[0, idx, :, :] # confidence map.

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (img_width * point[0]) / W
            y = (img_height * point[1]) / H

            if prob > self.threshold:
                points.append(x)
                points.append(y)
                points.append(prob)
            else:
                points.append(0)
                points.append(0)
                points.append(0)

        return points

def generate_pose_keypoints(img_file, pose_file):

    modelpath = 'pose'  # Adjust the model path accordingly
    pose_model = general_pose_model(modelpath)

    res_points = pose_model.predict(img_file)
    
    pose_data = {"version": 1,
                 "people":  [
                                {"pose_keypoints_2d": res_points}
                            ]
                }

    pose_keypoints_path = pose_file

    json_object = json.dumps(pose_data, indent=4) 
  
    # Writing to pose file
    with open(pose_keypoints_path, "w") as outfile: 
        outfile.write(json_object) 
    print('File saved at {}'.format(pose_keypoints_path))

generate_pose_keypoints(r"D:\vtryon_workout\M3D-VTON\images\person\front_test1-removebg-preview.png", r"D:\vtryon_workout\M3D-VTON\examples_3d\pose\front_test1-removebg-preview_keypoints.json")

#image depth
#----------------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np

# Load the JPG image
image = cv2.imread(r"D:\vtryon_workout\M3D-VTON\images\person\back_test1-removebg-preview.png")

# Convert the image to grayscale if it's not already in grayscale
if len(image.shape) == 3:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    grayscale_image = image

# Save the grayscale image as a .npy file
np.save(r"D:\vtryon_workout\M3D-VTON\examples_3d\depth\back_test1-removebg-preview.npy", grayscale_image)

print("Image converted and saved as output_image.npy")
