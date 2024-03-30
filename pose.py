import cv2
import numpy as np
import json

# Load the pretrained model and deploy prototxt file
net = cv2.dnn.readNetFromCaffe(r"D:\vtryon_workout\M3D-VTON\openpose_pose_coco.prototxt", r"D:\vtryon_workout\M3D-VTON\pose_iter_440000.caffemodel")

# Read input image
image = cv2.imread(r"D:\vtryon_workout\M3D-VTON\example\image\back_test1-removebg-preview.png")

# Prepare image for inference
width, height = image.shape[1], image.shape[0]
net.setInput(cv2.dnn.blobFromImage(image, 1.0, (width, height), (0, 0, 0), swapRB=False, crop=False))

# Forward pass through the network to get the output
output = net.forward()

# Convert float32 values to regular floats
output = output.astype(float)

# Parse the output to extract keypoints
keypoints = []
for i in range(0, output.shape[2]):
    # Extract confidence score
    confidence = output[0, 0, i, 2]
    if confidence > 0.01:  # Adjust threshold as needed
        # Key points location
        x = int(output[0, 0, i, 3] * width)
        y = int(output[0, 0, i, 4] * height)
        keypoints.append(x) 
        keypoints.append(y)
        keypoints.append(float(confidence)) # Convert confidence to float

# Convert keypoints to JSON format
json_data = {
    "version": 1.3,
    "people": [{
        "person_id": -1,
        "pose_keypoints_2d": keypoints,
        "face_keypoints_2d": [],  # You can add facial keypoints if needed
        "hand_left_keypoints_2d": [],  # You can add hand keypoints if needed
        "hand_right_keypoints_2d": []  # You can add hand keypoints if needed
    }]
}

# Save JSON data to a file
with open(r"D:\vtryon_workout\M3D-VTON\example\pose\back_test1-removebg-preview_keypoints.json", "w") as outfile:
    json.dump(json_data, outfile)


cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
