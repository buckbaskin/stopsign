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
        # TODO(buckbaskin): take a donut image and convert it to an unwrapped 360
        # degree image
        return donut

    def panorama_to_readings(self, panorama):
        # TODO(buckbaskin): given an unwrapped panorama, find blobs using opencv and
        #   convert their position in the image to a reading (bearing, size, color)
        list_of_readings = []
        list_of_readings.append(Reading(0, 1, 0, 0, 0, 0))
        return list_of_readings

    def has_stopsign(self, readings):
        return False
