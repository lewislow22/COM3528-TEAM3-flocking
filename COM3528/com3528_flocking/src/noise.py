#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports 2
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState, BatteryState, Image, Imu, Range, CompressedImage
from std_msgs.msg import Float32MultiArray, UInt32MultiArray, UInt16MultiArray, UInt8MultiArray, UInt16, UInt32, Int16MultiArray, String
from miro2_msg import msg

# MiRo-E modules and parameters
try:
	from . import miro_constants as con
except ImportError:
	import miro_constants as con
import miro2 as miro

# Other packages
import cv2
import datetime
import math
import numpy as np
import os
import rosnode
import rospy

# Imports
import os
import subprocess
from math import radians  # This is used to reset the head pose
import numpy as np  # Numerical Analysis library
import cv2  # Computer Vision library

import rospy  # ROS Python interface
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message
from sensor_msgs.msg import JointState  # ROS joints state message
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control) message

import miro2 as miro  # Import MiRo Developer Kit library

import miro_ros_interface as mri
import audio_perception as audio

miro_pub = mri.MiRoPublishers()
miro_pub.pub_tone(frequency=300, volume=20, duration=50)

class MiRoClient:
	
    rospy.Subscriber(self.tr + 'sensors/package', msg.sensors_package, self.callback_sensors)

    def __init__(self):
        # Initialise a new ROS node to communicate with MiRo, if needed
        rospy.init_node("noise", anonymous=True)
        # Give it some time to make sure everything is initialised
        rospy.sleep(2.0)
        # Initialise CV Bridge
        self.image_converter = CvBridge()
        # Individual robot name acts as ROS topic prefix
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        # Create two new subscribers to receive camera images with attached callbacks
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
        self.sub_mics = rospy.Subscriber(
            self.tr + 'sensors/mics',
            msg.sensors_package,
            self.callback_sensors
        )

        # Create a new publisher to send velocity commands to the robot
        self.vel_pub = rospy.Publisher(
            topic_base_name + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        # Create a new publisher to move the robot head
        self.pub_kin = rospy.Publisher(
            topic_base_name + "/control/kinematic_joints", JointState, queue_size=0
        )
        
        self.pub_kin = rospy.Publisher(
            topic_base_name + "/control/kinematic_joints", JointState, queue_size=0
        )

        # Create handle to store images
        self.input_camera = [None, None]
        # New frame notification
        self.new_frame = [False, False]
        # Create variable to store a list of ball's x, y, and r values for each camera
        self.ball = [None, None]
        # Set the default frame width (gets updated on receiving an image)
        self.frame_width = 640
        # Action selector to reduce duplicate printing to the terminal
        self.just_switched = True
        # Bookmark
        self.bookmark = 0
        # Move the head to default pose
        self.reset_head_pose()

    def loop(self):
        """
        Main control loop
        """
        print("Miro is listening")
        # Main control loop iteration counter
        self.counter = 0
        # This switch loops through MiRo behaviours:
        # Find ball, lock on to the ball and kick ball
        self.status_code = 0
        while not rospy.core.is_shutdown():

            # Step 1. Find ball
            if self.status_code == 1:
                # Every once in a while, look for ball
                if self.counter % self.CAM_FREQ == 0:
                    self.look_for_ball()

            # Step 2. Orient towards it
            elif self.status_code == 2:
                self.lock_onto_ball()

            # Step 3. Kick!
            elif self.status_code == 3:
                self.kick()

            # Fall back
            else:
                self.status_code = 1

            # Yield
            self.counter += 1
            rospy.sleep(self.TICK)

listener = audio.FrequencyPerception(300)
#print(listener.frequency_power([1000,800]))
print("test")

# This condition fires when the script is called directly
if __name__ == "__main__":
    main = MiRoClient()  # Instantiate class
    main.loop()  # Run the main control loop