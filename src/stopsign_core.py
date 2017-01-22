#!/usr/bin/env python

from collections import namedtuple
import cv2

Reading = namedtuple('Reading', ['bearing', 'size', 'r', 'g', 'b', 'a'])

class StopsignFinder(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0) # create video capture object

    def check_for_stopsign(self, unwrap=True, img=None):
        if img is None:
            img = self.retrieve_image()
        if unwrap:
            img = self.unwrap_image(img)
        readings = self.panorama_to_readings(img)
        return self.has_stopsign(readings)

    def retrieve_image(self):
        cv_image = self.cap.read()
        return cv_image

    def unwrap_image(self, donut):
        # TODO(buckbaskin): take a donut image and convert it to an unwrapped
        #   360 degree image
        return donut

    def panorama_to_readings(self, panorama):
        # TODO(buckbaskin): given an unwrapped panorama, find blobs using opencv
        #   and convert their position in the image to a reading (bearing, size,
        #   color)
        img = panorama

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

        # create a wrapping red filter (wraps from 160-20) (needs adjustment)
        low_sat = 75 # too grey'd out
        hi_sat = 255
        low_val = 30 # too dark
        hi_val = 205 # too bright

        lower_limit1 = np.array([160, low_sat, low_val])
        upper_limit1 = np.array([180, hi_sat, hi_val])
        lower_limit2 = np.array([0, low_sat, low_val])
        upper_limit2 = np.array([20, hi_sat, hi_val])
        mask1 = cv2.inRange(hsv_img, lower_limit1, upper_limit1)
        mask2 = cv2.inRange(hsv_img, lower_limit2, upper_limit2)
        mask_final = cv2.bitwise_or(mask1, mask2)
        mask_array.append(mask_final)
        res_array.append(cv2.bitwise_and(img, img, mask=mask_final))

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
            im = cv2.bitwise_not(mask_array[i])
            
            keypoints = detector.detect(im)
            keypoints_sum += len(keypoints)

            for blob in keypoints:
                blobs.append(blob_to_msg(img, blob))

        print('\nblob msgs '+str(len(blobs)))
        
        # TODO convert blobs to readings

        list_of_readings = []
        list_of_readings.append(Reading(0, 1, 0, 0, 0, 0))
        return list_of_readings

    def has_stopsign(self, readings):
        return False
