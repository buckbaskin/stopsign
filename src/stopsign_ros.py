import cv2
import rospy

from stopsign_core import StopsignFinder
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool

finder = StopsignFinder()

OUT = rospy.Publisher('stopsign', Bool, queue_size=1)

CAP = cv2.VideoCapture(0)

def active_image():
    cv_image = CAP.read()
    mess = Bool()
    mess.data = finder.check_for_stopsign(unwrap=False, img=cv_image, debug=False, save=False)
    OUT.publish(mess)

def image_cb(image_msg):
    header = image_msg.header
    h = image_msg.height
    w = image_msg.width
    encoding = image_msg.encoding
    is_bigendian = image_msg.is_bigendian
    uint32_step = image_msg.step
    image_data = image_msg.data

    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        mess = Bool()
        mess.data = finder.check_for_stopsign(unwrap=False, img=cv_image, debug=False, save=False)
        OUT.publish(mess)
    except CvBridgeError as cvbe:
        print(cvbe)
    
    return None

image_in = rospy.Subscriber('/camera/image/rgb', Image, image_cb)

if __name__ == '__main__':
    rospy.init_node('stopsign')
    while not rospy.is_shutdown():
        active_image()
