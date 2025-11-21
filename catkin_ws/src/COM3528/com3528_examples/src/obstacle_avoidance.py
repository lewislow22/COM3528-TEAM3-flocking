#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32MultiArray, UInt16
from sensor_msgs.msg import Range

import miro2 as miro
try:
    from miro2.lib import wheel_speed2cmd_vel
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel


class ObstacleAvoidance:
    def __init__(self, topic_root):

        self.pub_cmd_vel = rospy.Publisher(
            topic_root + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        self.vel = TwistStamped()

        self.sonar = 1.0
        self.cliff_left = 1.0
        self.cliff_right = 1.0
        self.light_fl = 0.0
        self.light_fr = 0.0

        rospy.Subscriber(topic_root + "/sensors/sonar", Range, self.sonar_cb)
        rospy.Subscriber(topic_root + "/sensors/cliff", UInt16, self.cliff_cb)
        rospy.Subscriber(topic_root + "/sensors/light", Float32MultiArray, self.light_cb)

        self.SONAR_MIN = 0.16
        self.SONAR_FAR = 0.35
        self.BACKUP_SPEED = -0.15
        self.TURN_SPEED = 0.40

        self.LIGHT_THRESHOLD = 0.75
        self.CLIFF_THRESHOLD = 0.2


    def sonar_cb(self, msg):
        self.sonar = msg.range

    def cliff_cb(self, msg):
        try:
            self.cliff_left = msg.data[0]
            self.cliff_right = msg.data[1]
        except:
            pass

    def light_cb(self, msg):
        self.light_fl = msg.data[0]
        self.light_fr = msg.data[1]


    def avoidance_required(self):
        reasons = []

        if self.sonar < self.SONAR_FAR:
            reasons.append("sonar")

        if self.cliff_left < self.CLIFF_THRESHOLD or self.cliff_right < self.CLIFF_THRESHOLD:
            reasons.append("cliff")

        if self.light_fl > self.LIGHT_THRESHOLD or self.light_fr > self.LIGHT_THRESHOLD:
            reasons.append("light_wall")

        return len(reasons) > 0, reasons


    def compute_avoidance(self):

        if self.cliff_left < self.CLIFF_THRESHOLD or self.cliff_right < self.CLIFF_THRESHOLD:
            if self.cliff_left < self.cliff_right:
                return self.BACKUP_SPEED, -self.BACKUP_SPEED
            else:
                return -self.BACKUP_SPEED, self.BACKUP_SPEED

        if self.sonar < self.SONAR_FAR:

            if self.sonar < self.SONAR_MIN:
                return self.BACKUP_SPEED, self.BACKUP_SPEED

            if self.light_fl > self.light_fr:
                return 0.0, +0.20
            else:
                return +0.20, 0.0

        if self.light_fl > self.LIGHT_THRESHOLD or self.light_fr > self.LIGHT_THRESHOLD:
            if self.light_fl > self.light_fr:
                return 0.0, +0.10
            else:
                return +0.10, 0.0

        return 0.0, 0.0


    def step(self, base_left, base_right):

        avoid_now, reasons = self.avoidance_required()

        if avoid_now:
            left, right = self.compute_avoidance()
        else:
            left, right = base_left, base_right

        (dr, dtheta) = wheel_speed2cmd_vel([left, right])
        self.vel.twist.linear.x = dr
        self.vel.twist.angular.z = dtheta
        self.pub_cmd_vel.publish(self.vel)

        return avoid_now, reasons

