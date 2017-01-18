'''
TODO:

Test Results:
- separate out gray/silver/blacks from colors, ignore bright whites
    - ended up ignoring them (avoids trying to use sky as feature for example)

- make overlapping color sets (range of 20, step of 10 kind of thing)
    - Need to eliminate overlapping/similar keypoints (see next)

- Check the effects of blurring image first (needs tuning)
    - blurring the image first creates a number of blobs that seem closer to my
        sense of what areas should be blobs
- Check the effects of blurring the masks (non-blurred image)
    - blurring the masks tends to group more points, but in some cases it splits
        up blobs that would otherwise have been together.
- Check the effects of blurring both...
    - blurs too much.
'''

import cv2
import math
import numpy as np

def pix_x_to_bearing(img, px):
    height, width, channels = img.shape
    return float(px)/float(width)*(2*math.pi)

def blob_to_msg(img, blob):
    from viz_feature_sim.msg import Blob
    b = Blob()
    b.size = blob.size
    b.bearing = pix_x_to_bearing(img, blob.pt[0])
    pixel = img[int(blob.pt[1])][int(blob.pt[0])]
    b.color.b = pixel[0]
    b.color.g = pixel[1]
    b.color.r = pixel[2]
    return b

def color_dist(blob_msga, blob_msgb):
    npa1 = np.array([[[blob_msga.color.b, blob_msga.color.g, blob_msga.color.r]]])
    npa2 = np.array([[[blob_msgb.color.b, blob_msgb.color.g, blob_msgb.color.r]]])
    npa1 = cv2.cvtColor(npa1, cv2.COLOR_BGR2HSV)
    npa2 = cv2.cvtColor(npa2, cv2.COLOR_BGR2HSV)
    h1 = int(npa1[0][0][0])
    h2 = int(npa2[0][0][0])
    return abs(h1 - h2)

img = cv2.imread('panorama3.jpg')
cv2.imshow('original image', img)
img = cv2.blur(img, (5,5,))

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask_array = []
res_array = []

# add in a gray-black filter
lower_limit = np.array([0,0,0])
upper_limit = np.array([180,75,255])
mask = cv2.inRange(hsv_img, lower_limit, upper_limit)
# mask_array.append(mask)
# res_array.append(cv2.bitwise_and(img, img, mask=mask))

# create a wrapping red filter (wraps from 170-10)
lower_limit1 = np.array([170,75,30])
upper_limit1 = np.array([180,255,205])
lower_limit2 = np.array([0,75,30])
upper_limit2 = np.array([10,255,205])
mask1 = cv2.inRange(hsv_img, lower_limit1, upper_limit1)
mask2 = cv2.inRange(hsv_img, lower_limit2, upper_limit2)
mask_final = cv2.bitwise_or(mask1, mask2)
mask_array.append(mask_final)
res_array.append(cv2.bitwise_and(img, img, mask=mask_final))

for i in range(0, 179, 10):
    # i-i+20 = hue/color ranges
    # 75-255 = saturation, colors that one can actually see
    # 50-205 = value/brightness, < 30 = gray, black, > 205 = white/sky
    lower_limit = np.array([i,75,30])
    upper_limit = np.array([i+20,255,205])
    mask = cv2.inRange(hsv_img, lower_limit, upper_limit)
    mask_array.append(mask)
    res_array.append(cv2.bitwise_and(img, img, mask=mask))

# find blobs in all the images!
# Set up the detector with default parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
# params.maxArea = 50000000
params.minArea = 500

# params.maxThreshold = 255
# params.minThreshold = 0

params.filterByConvexity = False
params.minConvexity = .00001
params.maxConvexity = 1.0

params.filterByCircularity = False
params.minCircularity = .00001
params.maxCircularity = 1.0

params.filterByColor = False

detector = cv2.SimpleBlobDetector(params)

keypoints_sum = 0

blobs = []

for i in range(0,len(mask_array)):
    if not i == 10:
        continue
    
    im = cv2.bitwise_not(mask_array[i])
    # im = cv2.blur(im, (5,5,))
    keypoints = detector.detect(im)
    keypoints_sum += len(keypoints)

    for blob in keypoints:
        blobs.append(blob_to_msg(img, blob))
        # print(blob_to_msg(img, blob).color)

    cv2.imshow('cut image '+str(i), res_array[i])
    cv2.imwrite('./blue_only.jpg', res_array[i])
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('Keypoints '+str(i), im_with_keypoints)
    cv2.imwrite('./mask_keypoints.jpg', im_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

filter_blobs = []
filter_blobs.append(blobs[0])

for i in range(0, len(blobs)):
    bloba = blobs[i]
    match = False
    for j in range(0, i):
        blobb = blobs[j]

        bearing_distance = abs(bloba.bearing - blobb.bearing)
        color_distance = color_dist(bloba, blobb)
        # print('%f x %f ...' % (color_distance, bearing_distance,))
        if bearing_distance < .1:
            if color_distance < 30:
                # print('filtered...')
                match = True
                break
    if not match:
        filter_blobs.append(bloba)

print('\nblob msgs '+str(len(blobs)))
print('\nf blob msgs '+str(len(filter_blobs)))