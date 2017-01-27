#!/usr/bin/env python

from collections import namedtuple
import cv2
import numpy as np

Reading = namedtuple('Reading', ['bearing', 'size', 'r', 'g', 'b', 'a'])

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
        self.params.minDistBetweenBlobs = 5.0

        self.detector = cv2.SimpleBlobDetector(self.params)


    def check_for_stopsign(self, unwrap=True, img=None, debug=False, save=False):
        if debug:
            print('1 type(img) -> %s' % (type(img),))
        if img is None:
            img = self.retrieve_image(debug=debug)
        if unwrap:
            img = self.unwrap_image(img, debug=debug)
        if debug:
            print('2 type(img) -> %s' % (type(img),))
        readings = self.panorama_to_readings(img, debug=debug, save=save)
        return self.has_stopsign(readings, debug=debug)

    def retrieve_image(self, debug=False):
        cv_image = self.cap.read()
        return cv_image

    def unwrap_image(self, donut, debug=False):
        # TODO(buckbaskin): take a donut image and convert it to an unwrapped
        #   360 degree image
        return donut

    def blob_detect(self, black_and_white, debug=False):
        im = black_and_white
        # Detect blobs.
        keypoints = self.detector.detect(im)
         
        if debug:
            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
             
            # Show keypoints
            cv2.imshow("Keypoints", im_with_keypoints)
            cv2.waitKey(0)

        return keypoints

    def panorama_to_readings(self, img, debug=False, save=False):
        # TODO(buckbaskin): given an unwrapped panorama, find blobs using opencv
        #   and convert their position in the image to a reading (bearing, size,
        #   color)
        img = cv2.blur(img, (5,5,))
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
        lower_limit = np.array([0,0,0])
        upper_limit = np.array([180,75,255])
        mask = cv2.inRange(hsv_img, lower_limit, upper_limit)
        # mask_array.append(mask)
        # res_array.append(cv2.bitwise_and(img, img, mask=mask))

        # create a wrapping red filter (wraps from 160-20) (needs adjustment)
        low_sat = 120 # too grey'd out
        hi_sat = 255
        low_val = 140 # too dark
        hi_val = 210 # too bright

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
        if save and isinstance(save, str):
            cv2.imwrite('processed/'+save+'_mask.jpg', res_array[-1])

        

        if debug:
            cv2.destroyAllWindows()
            cv2.waitKey()

        for i in range(0,len(mask_array)):
            img_not = cv2.bitwise_not(mask_array[i])
            
            keyp_not = self.blob_detect(img_not)

            new_keyp = []

            if debug:
                # img_with_keypoints = cv2.drawKeypoints(mask_array[i], keyp_mask, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # cv2.imshow('Positive with keyp', img_with_keypoints)
                # cv2.waitKey()
                # cv2.destroyAllWindows()
                cv2.imshow('img_not', img_not)
                cv2.waitKey()
                print('len(keyp_not): %d' % (len(keyp_not),))
                for point in keyp_not:
                    if point.size > 0.1:
                        new_keyp.append(point)
                not_with_keypoints = cv2.drawKeypoints(img_not, new_keyp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imshow('Negative with keyp', not_with_keypoints)
                cv2.waitKey()
                cv2.destroyAllWindows()
                import sys
                sys.exit(1)


            # new_keyp = []
            # for blob in keypoints:
            #     if blob.size > 10:
            #         new_keyp.append(blob)
            #         blobs.append(self.blob_to_rd(img, blob))

            # if debug:
            #     img_with_keypoints = cv2.drawKeypoints(img, new_keyp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #     cv2.imshow('Img Keypoints '+str(i), img_with_keypoints)
            # if save and isinstance(save, str):
            #     img_with_keypoints = cv2.drawKeypoints(img, new_keyp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #     cv2.imwrite('processed/'+save+'_keyp.jpg', img)
            # if debug:
            #     mask_with_keypoints = cv2.drawKeypoints(mask_array[i], new_keyp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #     cv2.imshow('Mask Keypoints '+str(i), mask_with_keypoints)
            # if debug:
            #     cv2.waitKey()
            #     cv2.destroyAllWindows()

        return blobs

    def blob_to_rd(self, img, blob, debug=False):
        return Reading(0, 1, 0, 0, 0, 0)

    def has_stopsign(self, readings, debug=False):
        return len(readings) > 1

