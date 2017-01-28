#!/usr/bin/env python

import rospy

from collections import namedtuple
import cv2
import numpy as np

from cv_bridge import CvBridge, CvBridgeError

Reading = namedtuple('Reading', ['bearing', 'size', 'r', 'g', 'b', 'a'])

COUNTER = 0

class StopsignFinder(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0) # create video capture object

        self.params = cv2.SimpleBlobDetector_Params()

        self.params.filterByConvexity = False
        self.params.filterByCircularity = False

        self.params.filterByColor = True
        self.params.minThreshold = 0
        self.params.maxThreshold = 50

        self.params.filterByArea = True
        self.params.minArea = 1000
        self.params.maxArea = 700000

        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.05
        self.params.minDistBetweenBlobs = 1.0

        self.detector = cv2.SimpleBlobDetector(self.params)

        self.bridge = CvBridge() # Create bridge between OpenCV and ROS


    def check_for_stopsign(self, img, unwrap=True, debug=False, save=False, pub_mask=None):
        if img is None:
            return False
        if debug:
            print('1 type(img) -> %s' % (type(img),))
        if unwrap:
            img = self.unwrap_image(img, debug=debug)
        if debug:
            print('2 type(img) -> %s' % (type(img),))
        readings = self.panorama_to_readings(img, debug=debug, save=save, pub_mask=pub_mask)
        return self.has_stopsign(readings, debug=debug)

    def retrieve_image(self, debug=False):
        cv_image = self.cap.read()
        return cv_image

    def unwrap_image(self, donut, debug=False):
        # TODO(buckbaskin): take a donut image and convert it to an unwrapped
        #   360 degree image
        return donut

    def blob_detect(self, black_and_white, debug=False):
        # Detect blobs.
        keypoints = self.detector.detect(black_and_white)

        if debug:
            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the
            # circle corresponds to the size of blob
            im_with_keypoints = cv2.drawKeypoints(
                black_and_white,
                keypoints,
                np.array([]),
                (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Show keypoints
            cv2.imshow("Keypoints", im_with_keypoints)
            cv2.waitKey(0)

        return keypoints

    def panorama_to_readings(self, img, debug=False, save=False, pub_mask=None):
        img = cv2.blur(img, (5, 5,))
        if save and isinstance(save, str):
            loc = './processed/%s_orig.jpg' % (save,)
            print('cv2.imwrite(%s, img)' % (loc,))
            cv2.imwrite(loc, img)
        if debug:
            cv2.imshow('blurred image', img)
            cv2.waitKey()

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_array = []
        res_array = []

        # add in a gray-black filter
        lower_limit = np.array([0, 0, 0])
        upper_limit = np.array([180, 75, 255])
        mask = cv2.inRange(hsv_img, lower_limit, upper_limit)
        # mask_array.append(mask)
        # res_array.append(cv2.bitwise_and(img, img, mask=mask))

        # create a wrapping red filter (wraps from 160-20) (needs adjustment)
        low_sat = 100 # too grey'd out
        hi_sat = 255
        low_val = 100 # too dark
        hi_val = 240 # too bright

        lower_limit1 = np.array([170, low_sat, low_val])
        upper_limit1 = np.array([180, hi_sat, hi_val])
        lower_limit2 = np.array([0, low_sat, low_val])
        upper_limit2 = np.array([10, hi_sat, hi_val])
        mask1 = cv2.inRange(hsv_img, lower_limit1, upper_limit1)
        mask2 = cv2.inRange(hsv_img, lower_limit2, upper_limit2)
        mask_final = cv2.bitwise_or(mask1, mask2)
        mask_final = cv2.dilate(mask_final, np.ones((5,5)), iterations=2)
        mask_array.append(mask_final)
        res_array.append(cv2.bitwise_and(img, img, mask=mask_final))
        if debug:
            cv2.imshow('mask', mask_array[-1])
            cv2.waitKey()
        if debug:
            cv2.imshow('result', res_array[-1])
            cv2.waitKey()
        if save and isinstance(save, str):
            cv2.imwrite('processed/'+save+'_mask.jpg', res_array[-1])

        if debug:
            cv2.destroyAllWindows()
            cv2.waitKey()

        for i in range(0, len(mask_array)):
            img_not = cv2.bitwise_not(mask_array[i])
            keyp_not = self.blob_detect(img_not)

            if pub_mask is not None:
                # not_with_keypoints = cv2.drawKeypoints(
                #     img_not,
                #     keyp_not,
                #     np.array([]),
                #     (0, 0, 255),
                #     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                rosImageMsg = self.bridge.cv2_to_imgmsg(img_not, encoding="mono8")
                # rosImageMsg.header.stamp = rospy.get_rostime()
                pub_mask.publish(rosImageMsg)

            if debug:
                cv2.imshow('img_not', img_not)
                cv2.waitKey()
                not_with_keypoints = cv2.drawKeypoints(
                    img_not,
                    keyp_not,
                    np.array([]),
                    (0, 0, 255),
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow('Negative with keyp', not_with_keypoints)
                cv2.waitKey()
                cv2.destroyAllWindows()

        return keyp_not

    def blob_to_rd(self, img, blob, debug=False):
        return Reading(0, 1, 0, 0, 0, 0)

    def has_stopsign(self, readings, debug=False):
        sorted(readings, key=lambda keyp: keyp.size)
        if len(readings) >= 2:
            readings = readings[-2:]

        size_accum = 0.0

        for keypoint in readings:
            size_accum += keypoint.size

        if size_accum > 45.0:
            if len(readings) == 2:
                x0 = readings[0].pt[0]
                x1 = readings[1].pt[0]

                if abs(x1 - x0) < 100:
                    # print('x1 - x0 %s' % (x1 - x0,))
                    global COUNTER
                    COUNTER += 1
                    print('%d Proper double blobs' % (COUNTER,))
                else:
                    if debug:
                        print('Non-vertical blobs')
                return abs(x1 - x0) < 100
            else:
                global COUNTER
                COUNTER += 1
                print('%d Big enough single blob' % (COUNTER,))
                return True
        else:
            if debug:
                print('Size Fail %s' % (size_accum,))
            return False
