#!/usr/bin/env python3

import os
import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32MultiArray, UInt16
from sensor_msgs.msg import Range, CompressedImage
import cv2

import miro2 as miro
try:
    from miro2.lib import wheel_speed2cmd_vel
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel


class ObstacleAvoidance:
    def __init__(self, topic_root):

        # Publisher for wheel speeds
        self.pub_cmd_vel = rospy.Publisher(
            topic_root + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        self.vel = TwistStamped()

        # Sensor values
        self.sonar = 1.0
        self.cliff_left = 1.0
        self.cliff_right = 1.0
        self.light_fl = 0.0
        self.light_fr = 0.0
        self.non_target_seen = False

        # Subscribers
        rospy.Subscriber(topic_root + "/sensors/sonar", Range, self.sonar_cb)
        rospy.Subscriber(topic_root + "/sensors/cliff", UInt16, self.cliff_cb)
        rospy.Subscriber(topic_root + "/sensors/light", Float32MultiArray, self.light_cb)
        rospy.Subscriber(topic_root + "/sensors/caml/compressed", CompressedImage, self.cam_cb)

        # Constants
        self.SONAR_MIN = 0.16
        self.SONAR_FAR = 0.35
        self.BACKUP_SPEED = -0.15
        self.LIGHT_THRESHOLD = 0.75
        self.CLIFF_THRESHOLD = 0.2
        self.AVOID_DURATION = 14

        # HSV green thresholds 
        self.COLOR_LOW = np.array([0, 180, 0])
        self.COLOR_HIGH = np.array([255, 255, 255])

        # Internal timer
        self.avoid_timer = 0

    # --- Sensor callbacks ---
    def sonar_cb(self, msg):
        self.sonar = msg.range

    def cliff_cb(self, msg):
        try:
            self.cliff_left = msg.data[0]
            self.cliff_right = msg.data[1]
        except Exception:
            pass

    def light_cb(self, msg):
        self.light_fl = msg.data[0]
        self.light_fr = msg.data[1]

    def cam_cb(self, msg):
        """Detect non-green objects in camera."""
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.non_target_seen = False
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.COLOR_LOW, self.COLOR_HIGH)
        green_pixels = np.sum(mask > 0)
        total_pixels = mask.size

        #  consider non-target if <30% of the frame is green
        self.non_target_seen = (green_pixels / total_pixels) < 0.30

    # --- Decision logic ---
    def avoidance_required(self):
        """Check if any avoidance condition is triggered."""
        if self.sonar < self.SONAR_FAR:
            return True
        if self.cliff_left < self.CLIFF_THRESHOLD or self.cliff_right < self.CLIFF_THRESHOLD:
            return True
        if self.light_fl > self.LIGHT_THRESHOLD or self.light_fr > self.LIGHT_THRESHOLD:
            return True
        if self.non_target_seen:
            return True
        return False

    def compute_avoidance(self):
        """Compute wheel speeds for avoidance based on priority."""

        # 1. Cliffs
        if self.cliff_left < self.CLIFF_THRESHOLD or self.cliff_right < self.CLIFF_THRESHOLD:
            if self.cliff_left < self.cliff_right:
                return self.BACKUP_SPEED, -self.BACKUP_SPEED
            else:
                return -self.BACKUP_SPEED, self.BACKUP_SPEED

        # 2. Sonar obstacle
        if self.sonar < self.SONAR_FAR:
            if self.sonar < self.SONAR_MIN:
                return self.BACKUP_SPEED, self.BACKUP_SPEED
            return 0.0, 0.25  # turn away

        # 3. Non-green visual object
        if self.non_target_seen:
            return 0.0, 0.25  # turn away slowly

        # 4. Default forward
        return 0.12, 0.12

    # --- Main loop step ---
    def step(self, base_left, base_right):
        avoiding = self.avoidance_required()

        if avoiding:
            self.avoid_timer = self.AVOID_DURATION

        if self.avoid_timer > 0:
            self.avoid_timer -= 1
            left, right = self.compute_avoidance()
        else:
            left, right = base_left, base_right

        dr, dtheta = wheel_speed2cmd_vel([left, right])
        self.vel.twist.linear.x = dr
        self.vel.twist.angular.z = dtheta
        self.pub_cmd_vel.publish(self.vel)

        return left, right


# --- testing ---
if __name__ == "__main__":
    rospy.init_node("obstacle_avoidance")
    topic_root = "/" + os.getenv("MIRO_ROBOT_NAME")

    oa = ObstacleAvoidance(topic_root)
    rate = rospy.Rate(50)

    # Move forward while avoiding non green obstacles
    while not rospy.is_shutdown():
        oa.step(0.12, 0.12)
        rate.sleep()
