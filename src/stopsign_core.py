#!/usr/bin/env python

from collections import namedtuple
import cv2
import numpy as np

Reading = namedtuple('Reading', ['bearing', 'size', 'r', 'g', 'b', 'a'])

class StopsignFinder(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0) # create video capture object

    def check_for_stopsign(self, unwrap=True, img=None, debug=False):
        print('1 type(img) -> %s' % (type(img),))
        if img is None:
            img = self.retrieve_image(debug=debug)
        if unwrap:
            img = self.unwrap_image(img, debug=debug)
        print('2 type(img) -> %s' % (type(img),))
        readings = self.panorama_to_readings(img, debug=debug)
        return self.has_stopsign(readings, debug=debug)

    def retrieve_image(self, debug=False):
        cv_image = self.cap.read()
        return cv_image

    def unwrap_image(self, donut, debug=False):
        # TODO(buckbaskin): take a donut image and convert it to an unwrapped
        #   360 degree image
        return donut

    def panorama_to_readings(self, img, debug=False):
        # TODO(buckbaskin): given an unwrapped panorama, find blobs using opencv
        #   and convert their position in the image to a reading (bearing, size,
        #   color)
        img = cv2.blur(img, (5,5,))
        if debug:
            cv2.imshow('blurred image', img)
            cv2.waitKey()

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
        low_sat = 200 # too grey'd out
        hi_sat = 255
        low_val = 30 # too dark
        hi_val = 205 # too bright

        lower_limit1 = np.array([170, low_sat, low_val])
        upper_limit1 = np.array([180, hi_sat, hi_val])
        lower_limit2 = np.array([0, low_sat, low_val])
        upper_limit2 = np.array([10, hi_sat, hi_val])
        mask1 = cv2.inRange(hsv_img, lower_limit1, upper_limit1)
        mask2 = cv2.inRange(hsv_img, lower_limit2, upper_limit2)
        mask_final = cv2.bitwise_or(mask1, mask2)
        mask_array.append(mask_final)
        res_array.append(cv2.bitwise_and(img, img, mask=mask_final))
        if debug:
            cv2.imshow('mask', mask_array[-1])
            cv2.waitKey()
        if debug:
            cv2.imshow('result', res_array[-1])
            cv2.waitKey()

        # find blobs in all the images!
        # Set up the detector with default parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = False
        # params.maxArea = 50000000
        params.minArea = 100

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
            img = cv2.bitwise_not(mask_array[i])
            
            keypoints = detector.detect(img)

            cv2.destroyAllWindows()
            cv2.waitKey()

            new_keyp = []
            for blob in keypoints:
                if blob.size > 10:
                    new_keyp.append(blob)
                    blobs.append(self.blob_to_rd(img, blob))

            if debug:
                img_with_keypoints = cv2.drawKeypoints(img, new_keyp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow('Img Keypoints '+str(i), img_with_keypoints)
                mask_with_keypoints = cv2.drawKeypoints(mask_array[i], new_keyp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow('Mask Keypoints '+str(i), mask_with_keypoints)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return blobs

    def blob_to_rd(self, img, blob, debug=False):
        return Reading(0, 1, 0, 0, 0, 0)

    def has_stopsign(self, readings, debug=False):
        return len(readings) > 1

