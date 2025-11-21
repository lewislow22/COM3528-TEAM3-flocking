#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script makes MiRo look for a blue ball and stop once it reaches it
"""

# Imports
import os
from math import radians
import numpy as np
import cv2

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TwistStamped

import miro2 as miro

try:  # Python 3
    from miro2.lib import wheel_speed2cmd_vel
except ImportError:  # Python 2
    from miro2.utils import wheel_speed2cmd_vel


class MiRoClient:
    """
    Main MiRo controller class
    """
    TICK = 0.02
    CAM_FREQ = 1
    SLOW = 0.1
    DEBUG = False

    PREPROCESSING_ORDER = ["edge", "smooth", "color", "gaussian"]
    HSV = True
    f = lambda x: int(0) if (x < 0) else (int(255) if x > 255 else int(x))
    COLOR_HSV = [f(255), f(0), f(0)]
    COLOR_LOW = (f(180), f(0), f(0))
    COLOR_HIGH = (f(255), f(255), f(255))

    INTENSITY_LOW = 50
    INTENSITY_HIGH = 50

    GAUSSIAN_BLURRING = False
    KERNEL_SIZE = 15
    STANDARD_DEVIATION = 0
    DIFFERENCE_SD_LOW = 1.5
    DIFFERENCE_SD_HIGH = 0

    def reset_head_pose(self):
        self.kin_joints = JointState()
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, radians(34.0), 0.0, 0.0]
        t = 0
        while not rospy.core.is_shutdown():
            self.pub_kin.publish(self.kin_joints)
            rospy.sleep(self.TICK)
            t += self.TICK
            if t > 1:
                break

    def drive(self, speed_l=0.0, speed_r=0.0):
        """
        Drive MiRo by converting wheel speeds to cmd_vel
        """
        msg_cmd_vel = TwistStamped()
        wheel_speed = [speed_l, speed_r]
        (dr, dtheta) = wheel_speed2cmd_vel(wheel_speed)
        msg_cmd_vel.twist.linear.x = dr
        msg_cmd_vel.twist.angular.z = dtheta
        self.vel_pub.publish(msg_cmd_vel)

    def callback_caml(self, ros_image):
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image):
        self.callback_cam(ros_image, 1)

    def callback_cam(self, ros_image, index):
        try:
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.input_camera[index] = image
            self.frame_height, self.frame_width, channels = image.shape
            self.x_centre = self.frame_width / 2.0
            self.y_centre = self.frame_height / 2.0
            self.new_frame[index] = True
        except CvBridgeError:
            pass

    def detect_ball(self, frame, index):
        if frame is None:
            return

        if self.DEBUG:
            cv2.imshow("camera" + str(index), frame)
            cv2.waitKey(1)

        self.new_frame[index] = False
        processed_img = frame

        for method in self.PREPROCESSING_ORDER:
            if method == "color":
                if self.HSV:
                    im_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    rgb_colour = np.uint8([[self.COLOR_HSV]])
                    hsv_colour = cv2.cvtColor(rgb_colour, cv2.COLOR_RGB2HSV)
                    target_hue = hsv_colour[0, 0][0]
                    hsv_lo_end = np.array([target_hue - 20, 70, 70])
                    hsv_hi_end = np.array([target_hue + 20, 255, 255])
                    mask = cv2.inRange(im_hsv, hsv_lo_end, hsv_hi_end)
                    processed_img = cv2.bitwise_and(processed_img, processed_img, mask=mask)
                else:
                    mask = cv2.inRange(frame, self.COLOR_LOW, self.COLOR_HIGH)
                    processed_img = cv2.bitwise_and(processed_img, processed_img, mask=mask)

            elif method == "gaussian":
                sigma1 = self.DIFFERENCE_SD_LOW
                sigma2 = self.DIFFERENCE_SD_HIGH
                img_gauss1 = cv2.GaussianBlur(processed_img, (0, 0), sigma1)
                img_gauss2 = cv2.GaussianBlur(processed_img, (0, 0), sigma2)
                processed_img = img_gauss1 - img_gauss2

            elif method == "smooth":
                kernel_size = (self.KERNEL_SIZE, self.KERNEL_SIZE)
                if not self.GAUSSIAN_BLURRING:
                    kernel = np.ones(kernel_size, np.float32) / kernel_size[0]**2
                    processed_img = cv2.filter2D(processed_img, -1, kernel)
                else:
                    sigma = self.STANDARD_DEVIATION
                    processed_img = cv2.GaussianBlur(processed_img, (0, 0), sigma)

            elif method == "edge":
                processed_img = cv2.Canny(processed_img, self.INTENSITY_LOW, self.INTENSITY_HIGH)

        if len(processed_img.shape) == 3:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            processed_img,
            cv2.HOUGH_GRADIENT,
            1,
            40,
            param1=10,
            param2=10,
            minRadius=5,
            maxRadius=50,
        )

        if circles is None:
            return

        max_circle = None
        self.max_rad = 0
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            if c[2] > self.max_rad:
                self.max_rad = c[2]
                max_circle = c
        if max_circle is None:
            return

        cv2.circle(frame, (max_circle[0], max_circle[1]), max_circle[2], (0, 255, 0), 2)
        cv2.circle(frame, (max_circle[0], max_circle[1]), 2, (0, 0, 255), 3)
        if self.DEBUG:
            cv2.imshow("circles" + str(index), frame)
            cv2.waitKey(1)

        max_circle = np.array(max_circle).astype("float32")
        max_circle[0] -= self.x_centre
        max_circle[0] /= self.frame_width
        max_circle[1] -= self.y_centre
        max_circle[1] /= self.frame_width
        max_circle[1] *= -1.0
        max_circle[2] /= self.frame_width

        return [max_circle[0], max_circle[1], max_circle[2]]

    def look_for_ball(self):
        if self.just_switched:
            print("MiRo is looking for the ball...")
            self.just_switched = False
        for index in range(2):
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            self.ball[index] = self.detect_ball(image, index)
        if not self.ball[0] and not self.ball[1]:
            self.drive(self.SLOW, -self.SLOW)
        else:
            self.status_code = 2
            self.just_switched = True

    def lock_onto_ball(self, error=25):
        if self.just_switched:
            print("MiRo is locking on to the ball")
            self.just_switched = False
        for index in range(2):
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            self.ball[index] = self.detect_ball(image, index)

        if not self.ball[0] and self.ball[1]:
            self.drive(self.SLOW, -self.SLOW)
        elif self.ball[0] and not self.ball[1]:
            self.drive(-self.SLOW, self.SLOW)
        elif self.ball[0] and self.ball[1]:
            error = 0.05
            left_x = self.ball[0][0]
            right_x = self.ball[1][0]
            rotation_speed = 0.03
            if abs(left_x) - abs(right_x) > error:
                self.drive(rotation_speed, -rotation_speed)
            elif abs(left_x) - abs(right_x) < -error:
                self.drive(-rotation_speed, rotation_speed)
            else:
                self.status_code = 3  # Switch to stopping
                self.just_switched = True
        else:
            self.status_code = 0
            print("MiRo has lost the ball...")
            self.just_switched = True

    def kick(self):
        """
        [3 of 3] Stop at the ball instead of kicking
        """
        if self.just_switched:
            print("MiRo reached the ball, stopping!")
            self.just_switched = False
        self.drive(0.0, 0.0)
        self.status_code = 0
        self.just_switched = True

    def __init__(self):
        rospy.init_node("stop_at_ball", anonymous=True)
        rospy.sleep(2.0)
        self.image_converter = CvBridge()
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME", "miro")
        self.sub_caml = rospy.Subscriber(
            topic_base_name + "/sensors/caml/compressed",
            CompressedImage,
            self.callback_caml,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.sub_camr = rospy.Subscriber(
            topic_base_name + "/sensors/camr/compressed",
            CompressedImage,
            self.callback_camr,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        self.pub_kin = rospy.Publisher(
            topic_base_name + "/control/kinematic_joints", JointState, queue_size=0
        )
        self.input_camera = [None, None]
        self.new_frame = [False, False]
        self.ball = [None, None]
        self.frame_width = 640
        self.just_switched = True
        self.bookmark = 0
        self.reset_head_pose()

    def loop(self):
        print("MiRo looking for the ball and stopping at it. Press CTRL+C to halt...")
        self.counter = 0
        self.status_code = 0
        while not rospy.core.is_shutdown():
            if self.status_code == 1:
                if self.counter % self.CAM_FREQ == 0:
                    self.look_for_ball()
            elif self.status_code == 2:
                self.lock_onto_ball()
            elif self.status_code == 3:
                self.kick()
            else:
                self.status_code = 1
            self.counter += 1
            rospy.sleep(self.TICK)


if __name__ == "__main__":
    main = MiRoClient()
    main.loop()
