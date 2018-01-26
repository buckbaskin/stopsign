#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
from std_msgs.msg import String

import cv2
import numpy as np
import time
import datetime
from threading import Thread

# here is opencv setting code
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
 
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
 
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
 
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
 
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
 
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()

 
class WebcamVideoStream:
	def __init__(self, src):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 400.0)
		self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 300.0)
		
 
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		Thread(target=self.update, args=()).start()
		return self
 
	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return
 
			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
 
	def read(self):
		# return the frame most recently read
		return self.frame
 
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True



# set HSV color space parameter
H_MIN = 6
H_MAX = 160

H_ALL_MIN = 0
H_ALL_MAX = 180

S_MIN = 80
S_MAX = 255

V_MIN = 50
V_MAX = 255

lower_boundry = np.array([H_MIN ,S_MIN ,V_MIN])
upper_boundry = np.array([H_MAX ,S_MAX ,V_MAX])

lower_boundry_all = np.array([H_ALL_MIN ,S_MIN ,V_MIN])
upper_boundry_all = np.array([H_ALL_MAX ,S_MAX ,V_MAX])

# set cascade


#stop_sign_cascade = cv2.CascadeClassifier('stopsign_classifier.xml')
stop_sign_cascade = cv2.CascadeClassifier('/home/stopsign_classifier.xml')

#cap = cv2.VideoCapture(0)
vs = WebcamVideoStream(src=0).start()

t0 = 0
erodekernel = np.ones((3,3), np.uint8)
dilatekernel = np.ones((32,32), np.uint8)
#setting end
 
# test start

pub = rospy.Publisher('stop_sign', String, queue_size=10)
rospy.init_node('ss_detector', anonymous=True)
#rate = rospy.Rate(1) # 1hz
while not rospy.is_shutdown():
    # opencv code
    print (1.0/(time.time() - t0))
    t0 = time.time()
    #ret, img = cap.read()  
    img = vs.read()
    #if img == None: 
    #   raise Exception("could not load image !")

    t1 = time.time()  

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_not_red = cv2.inRange(hsv, lower_boundry, upper_boundry)
    mask_all = cv2.inRange(hsv, lower_boundry_all, upper_boundry_all)
    mask = mask_all - mask_not_red    

    mask = cv2.erode(mask, erodekernel, iterations=1)
    mask = cv2.dilate(mask, dilatekernel, iterations=1)
    res = cv2.bitwise_and(img,img, mask= mask)

    t2 = time.time() 

    #print ("preprocess: " + str(t2 - t1))

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)    
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    

    stop_sign = stop_sign_cascade.detectMultiScale(gray) 

    if (len(stop_sign)):
        #print(' there is a stop sign ')
        # ros output
        result_str = "yes" 
        rospy.loginfo(result_str)
        pub.publish(result_str)
        #rate.sleep()
        # end
    else:
        result_str = "no" 
        rospy.loginfo(result_str)
        pub.publish(result_str)

    t3 = time.time()
    #print("cascade: " + str(t3 - t2)) 
        
    #for (x,y,w,h) in stop_sign:
    #    cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
    cv2.imshow('res', res)  
    #cv2.imshow('gray', gray)    
    cv2.imshow('img', img)     

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        cv2.destroyAllWindows()
        break       

    
vs.stop()
        
