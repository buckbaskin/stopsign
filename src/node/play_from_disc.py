#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
imPub = rospy.Publisher(topic_out,Image,queue_size=1)

base = 'frame%04d.jpg'
for i in range(0, 900):
    print(i)
    cvImage = cv2.imread(base % i)
    if cvImage is not None:
        rosImageMsg = self.bridge.cv2_to_imgmsg(cvImage[1],encoding="passthrough")
        rosImageMsg.header.stamp = rospy.get_rostime()
        imPub.publish(rosImageMsg)

print('done!')