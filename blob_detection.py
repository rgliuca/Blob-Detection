import cv2
import numpy as np

# Read image
im=cv2.imread("images\\blob.jpg", cv2.IMREAD_GRAYSCALE)

im = cv2.GaussianBlur(im, (5, 5), 0)
#im = cv2.bilateralFilter(im,15,75,75)

# Set up the detector with default parameters.
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 150;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 2000
#params.maxArea = 10000

 
# Filter by Circularity
params.filterByCircularity = True
#params.minCircularity = 0.1
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = .8
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
 
detector = cv2.SimpleBlobDetector_create(params) 

# Detect blobs.
keypoints = detector.detect(im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
