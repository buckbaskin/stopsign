import cv2
import numpy as np;
 
# Read image
im = cv2.imread("images/blob3.jpg", cv2.IMREAD_GRAYSCALE)
 
params = cv2.SimpleBlobDetector_Params()

params.filterByConvexity = False
params.filterByCircularity = False

params.filterByColor = True
params.minThreshold = 0
params.maxThreshold = 50

params.filterByArea = True
params.minArea = 100
params.maxArea = 700000

params.filterByInertia = True
params.minInertiaRatio = 0.05
print('inertia')
print(params.minInertiaRatio)
print(params.maxInertiaRatio)

params.minDistBetweenBlobs = 1.0

# for attr in ['blobColor', 'filterByArea', 'filterByCircularity', 'filterByColor', 'filterByConvexity', 'filterByInertia', 'maxArea', 'maxCircularity', 'maxConvexity', 'maxInertiaRatio', 'maxThreshold', 'minArea', 'minCircularity', 'minConvexity', 'minDistBetweenBlobs', 'minInertiaRatio', 'minRepeatability', 'minThreshold', 'thresholdStep']:
#     print(attr)
#     print(getattr(params, attr))

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector(params)
 
# Detect blobs.
keypoints = detector.detect(im)
 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
