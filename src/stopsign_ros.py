import cv2
import stopsign_core
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from viz_feature_sim.msg import VizScan, Blob

bearing_out = rospy.Publisher('processed_image/bearings', VizScan, queue_size=1)
bridge = CvBridge()

def core_to_rosmsg(reading_list):
    temp = []
    for reading in reading_list:
        blob = Blob()
        blob.bearing = reading.bearing
        blob.size = reading.size
        blob.color.r = reading.r
        blob.color.g = reading.g
        blob.color.b = reading.b
        temp.append(blob)
    return temp

def image_cb(image_msg):
    header = image_msg.header
    h = image_msg.height
    w = image_msg.width
    encoding = image_msg.encoding
    is_bigendian = image_msg.is_bigendian
    uint32 step = image_msg.step
    image_data = image_msg.data

    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        viz_scan = VizScan()
        bearing_out.publish(core_to_rosmsg(stopsign_core.one_click_process()))
    except CvBridgeError as cvbe:
        print cvbe
        return None



image_in = rospy.Subscriber('/camera/image/rgb', Image, image_cb)

if __name__ == '__main__':
    rospy.init_node('stopsign')
    rospy.spin()
