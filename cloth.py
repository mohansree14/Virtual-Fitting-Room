import cv2
from PIL import Image
org_path1=r"D:\vtryon_workout\M3D-VTON\example\cloth\shirt_back_test_1.PNG"
org_path2=r"D:\vtryon_workout\M3D-VTON\example\cloth\shirt_front_test1.PNG"
cloth_1=cv2.imread(org_path1)
cloth_2=cv2.imread(org_path2)
def resize_img(path):
    c = Image.open(path)
    c = c.resize((320, 512), Image.BICUBIC).convert('RGB')
    c.save(path)
cv2.imwrite(r"D:\vtryon_workout\M3D-VTON\example\cloth\shirt_back_test_1.PNG",cloth_1)
resize_img(r"D:\vtryon_workout\M3D-VTON\example\cloth\shirt_back_test_1.PNG")
cv2.imwrite(r"D:\vtryon_workout\M3D-VTON\example\cloth\shirt_front_test1.PNG",cloth_2)
resize_img(r"D:\vtryon_workout\M3D-VTON\example\cloth\shirt_front_test1.PNG")

resize_img(r"D:\vtryon_workout\M3D-VTON\example\image\back_test1-removebg-preview.png")
resize_img(r"D:\vtryon_workout\M3D-VTON\example\image\front_test1-removebg-preview.png")
#mask

cloth=cv2.imread(r"D:\vtryon_workout\M3D-VTON\example\cloth\shirt_front_test1.PNG")
import cv2
import numpy as np

# Read the image
image = cloth

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary mask
_, mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

# Invert the mask
mask = 255 - mask

# Display the mask
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the mask
cv2.imwrite(r"D:\vtryon_workout\M3D-VTON\example\cloth-mask\shirt_front_test1.PNG", mask)




