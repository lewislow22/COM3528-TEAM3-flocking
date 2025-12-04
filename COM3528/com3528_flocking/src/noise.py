#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy  # ROS Python interface
import miro_ros_interface as mri

miro_pub = mri.MiRoPublishers()

# This condition fires when the script is called directly
if __name__ == "__main__":
    while not rospy.core.is_shutdown():
    miro_pub.pub_tone(frequency=800, volume=100, duration=3)
    rospy.sleep(0.2)
    miro_pub.pub_tone(frequency=200, volume=0, duration=3)
    rospy.sleep(3)
