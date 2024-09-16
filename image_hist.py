import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
  
# reading the input image 
img = cv2.imread('mountain.jpg') 
  
# define colors to plot the histograms 
colors = ('b','g','r') 
  
# compute and plot the image histograms 
for i,color in enumerate(colors): 
    hist = cv2.calcHist([img],[i],None,[256],[0,256]) 
    
    plt.plot(hist,color = color) 
plt.title('Image Histogram GFG') 
plt.show()


image = Image.open('path_to_image.jpg').convert('L')  # 'L' mode is grayscale
image_np = np.array(image)

======
# Step 2: Compute histogram using NumPy
hist, bins = np.histogram(image_np, bins=256, range=(0, 255))
=====

image_np = np.array(image)

# Step 2: Compute histograms for each color channel
colors = ('Red', 'Green', 'Blue')
color_histograms = {}

for i, color in enumerate(colors):
    # Flatten each color channel and calculate histogram
    hist, bins = np.histogram(image_np[:, :, i], bins=256, range=(0, 255))
    color_histograms[color] = hist