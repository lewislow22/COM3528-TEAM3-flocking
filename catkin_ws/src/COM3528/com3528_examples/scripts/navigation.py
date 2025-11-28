#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script makes MiRo look for a blue ball and kick it

The code was tested for Python 2 and 3
For Python 2 you might need to change the shebang line to
#!/usr/bin/env python
"""

# Imports
import os
import subprocess
from math import radians  # This is used to reset the head pose
import math
import numpy as np  # Numerical Analysis library
import cv2  # Computer Vision library

import rospy  # ROS Python interface
from sensor_msgs.msg import CompressedImage  # ROS CompressedImage message
from sensor_msgs.msg import JointState  # ROS joints state message
from cv_bridge import CvBridge, CvBridgeError  # ROS -> OpenCV converter
from geometry_msgs.msg import TwistStamped  # ROS cmd_vel (velocity control) message

from obstacle_avoidance import ObstacleAvoidance

import miro2 as miro  # Import MiRo Developer Kit library

try:  # For convenience, import this util separately
    from miro2.lib import wheel_speed2cmd_vel  # Python 3
except ImportError:
    from miro2.utils import wheel_speed2cmd_vel  # Python 2

class MiRoClient:
    """
    Script settings below
    """
    ##########################
    TICK = 0.02  # This is the update interval for the main control loop in secs
    CAM_FREQ = 1  # Number of ticks before camera gets a new frame, increase in case of network lag
    SLOW = 0.1  # Radial speed when turning on the spot (rad/s)
    FAST = 0.4  # Linear speed when kicking the ball (m/s)
    DEBUG = True # Set to True to enable debug views of the cameras
    TRANSLATION_ONLY = False # Whether to rotate only
    IS_MIROCODE = False  # Set to True if running in MiRoCODE

    # formatting order
    PREPROCESSING_ORDER = ["edge", "smooth", "color", "gaussian"]
        # set to empty to not preprocess or add the methods in the order you want to implement.
        # "edge" to use edge detection, "gaussian" to use difference gaussian
        # "color" to use color segmentation, "smooth" to use smooth blurring,

    # color segmentation format
    HSV = True  # if true select a color which will convert to hsv format with a range of its own, else you can select your own rgb range
    f = lambda x: int(0) if (x < 0) else (int(255) if x > 255 else int(x))
    COLOR_HSV = [f(255), f(0), f(0)]     # target color which will be converted to hsv for processing, format BGR
    COLOR_LOW = (f(180), f(0), f(0))         # low color segment, format BGR
    COLOR_HIGH = (f(255), f(255), f(255))  # high color segment, format BGR

    # edge detection format
    INTENSITY_LOW = 50   # min 0, max 500
    INTENSITY_HIGH = 50  # min 0, max 500

    # smoothing_blurring
    GAUSSIAN_BLURRING = False
    KERNEL_SIZE = 15         # min 3, max 15
    STANDARD_DEVIATION = 0  # min 0.1, max 4.9

    # difference gaussian
    DIFFERENCE_SD_LOW = 1.5 # min 0.00, max 1.40
    DIFFERENCE_SD_HIGH = 0 # min 0.00, max 1.40
    ##########################
    """
    End of script settings
    """

    def reset_head_pose(self):
        """
        Reset MiRo head to default position, to avoid having to deal with tilted frames
        """
        self.kin_joints = JointState()  # Prepare the empty message
        self.kin_joints.name = ["tilt", "lift", "yaw", "pitch"]
        self.kin_joints.position = [0.0, radians(34.0), 0.0, 0.0]
        t = 0
        while not rospy.core.is_shutdown():  # Check ROS is running
            # Publish state to neck servos for 1 sec
            self.pub_kin.publish(self.kin_joints)
            rospy.sleep(self.TICK)
            t += self.TICK
            if t > 1:
                break
        self.INTENSITY_CHECK = lambda x: int(0) if (x < 0) else (int(500) if x > 500 else int(x))
        self.KERNEL_SIZE_CHECK = lambda x: int(3) if (x < 3) else (int(15) if x > 15 else int(x))
        self.STANDARD_DEVIATION_PROCESS = lambda x: 0.1 if (x < 0.1) else (4.9 if x > 4.9 else round(x, 1))
        self.DIFFERENCE_CHECK = lambda x: 0.01 if (x < 0.01) else (1.40 if x > 1.40 else round(x,2))

    def drive(self, speed_l=0.1, speed_r=0.1):
        """
        Simple drive function: send wheel speeds directly without obstacle avoidance.
        """
        cmd_vel = wheel_speed2cmd_vel(speed_l, speed_r)
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
        pub = rospy.Publisher(topic_base_name + "/cmd_vel", TwistStamped, queue_size=1)
        pub.publish(cmd_vel)
        # Update heading estimate
        self.update_heading_estimate(speed_l, speed_r)


    def callback_caml(self, ros_image):  # Left camera
        self.callback_cam(ros_image, 0)

    def callback_camr(self, ros_image):  # Right camera
        self.callback_cam(ros_image, 1)

    def callback_cam(self, ros_image, index):
        """
        Callback function executed upon image arrival
        """
        # Silently(-ish) handle corrupted JPEG frames
        try:
            # Convert compressed ROS image to raw CV image
            image = self.image_converter.compressed_imgmsg_to_cv2(ros_image, "rgb8")
            # Convert from OpenCV's default BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Store image as class attribute for further use
            self.input_camera[index] = image
            # Get image dimensions
            self.frame_height, self.frame_width, channels = image.shape
            self.x_centre = self.frame_width / 2.0
            self.y_centre = self.frame_height / 2.0
            # Raise the flag: A new frame is available for processing
            self.new_frame[index] = True
        except CvBridgeError as e:
            # Ignore corrupted frames
            pass

    def detect_ball(self, frame, index):
        """
        Detect a small ball in a given camera frame using dynamic HSV color settings.
        Returns normalized [x, y, r] of the largest circle detected, or None if not found.
        """

        self.COLOR_HSV = [0, 0, 255]   # Blue in RGB (correct)

        if frame is None:
            return None

        # Flag this frame as processed
        self.new_frame[index] = False

        # Copy frame for processing
        processed_img = frame.copy()

        # ------------------------
        # 1. Preprocessing: smoothing if requested
        # ------------------------
        if "smooth" in self.PREPROCESSING_ORDER:
            ksize = self.KERNEL_SIZE_CHECK(self.KERNEL_SIZE)
            processed_img = cv2.GaussianBlur(processed_img, (ksize, ksize), 0)

        # ------------------------
        # 2. Convert to HSV
        # ------------------------
        im_hsv = cv2.cvtColor(processed_img, cv2.COLOR_RGB2HSV)

        # Dynamically compute HSV range from COLOR_HSV
        rgb_target = np.uint8([[self.COLOR_HSV]])  # [[R,G,B]]
        hsv_target = cv2.cvtColor(rgb_target, cv2.COLOR_RGB2HSV)[0][0]
        hue = int(hsv_target[0])
        # Use ±20 hue range, standard S/V thresholds
        hsv_low = np.array([max(hue-20, 0), 70, 70])
        hsv_high = np.array([min(hue+20, 179), 255, 255])

        # ------------------------
        # 3. Mask by color
        # ------------------------
        mask = cv2.inRange(im_hsv, hsv_low, hsv_high)
        masked_img = cv2.bitwise_and(processed_img, processed_img, mask=mask)

        # Debug: show masks
        if self.DEBUG:
            cv2.imshow(f"mask{index}", mask)
            cv2.imshow(f"masked_img{index}", masked_img)
            cv2.waitKey(1)

        # ------------------------
        # 4. Convert to grayscale for circle detection
        # ------------------------
        gray = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)
        if self.DEBUG:
            cv2.imshow(f"gray{index}", gray)
            cv2.waitKey(1)

        # ------------------------
        # 5. Circle detection
        # ------------------------
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,
            param1=50,
            param2=15,
            minRadius=5,
            maxRadius=50
        )

        if circles is None:
            return None

        circles = np.uint16(np.around(circles))
        max_circle = max(circles[0, :], key=lambda c: c[2])  # largest radius

        # Draw circle for debugging
        if self.DEBUG:
            cv2.circle(frame, (max_circle[0], max_circle[1]), max_circle[2], (0, 255, 0), 2)
            cv2.circle(frame, (max_circle[0], max_circle[1]), 2, (0, 0, 255), 3)
            cv2.imshow(f"circles{index}", frame)
            cv2.waitKey(1)

        # ------------------------
        # 6. Normalize coordinates
        # ------------------------
        x_norm = (max_circle[0] - self.frame_width / 2) / self.frame_width
        y_norm = -(max_circle[1] - self.frame_height / 2) / self.frame_width  # invert y
        r_norm = max_circle[2] / self.frame_width

        return [x_norm, y_norm, r_norm]

    def look_for_ball(self):
        """
        Look around to detect the ball.
        Rotate slowly if not seen.
        """
        ball_seen = False
        for index in range(2):
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            self.ball[index] = self.detect_ball(image, index)
            if self.ball[index]:
                self.last_seen_side = index
                ball_seen = True

        if ball_seen:
            self.status_code = 2  # Switch to lock-on
            self.just_switched = True
        else:
            # Rotate slowly left/right until found
            self.drive(self.SLOW, -self.SLOW)



    def lock_onto_ball(self, alignment_threshold=0.05, radius_threshold=0.15):
        """
        Turn MiRo to face the ball and move forward until close enough.
        """
        for index in range(2):
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            self.ball[index] = self.detect_ball(image, index)

        left_ball = self.ball[0]
        right_ball = self.ball[1]

        if left_ball or right_ball:
            # Simple alignment: choose whichever sees the ball
            x = left_ball[0] if left_ball else right_ball[0]

            if abs(x) > alignment_threshold:
                # Turn toward the ball
                turn_speed = self.SLOW
                if x > 0:
                    self.drive(turn_speed, -turn_speed)  # clockwise
                else:
                    self.drive(-turn_speed, turn_speed)  # counter-clockwise
            else:
                # Aligned: move forward
                ball_radius = left_ball[2] if left_ball else right_ball[2]
                if ball_radius < radius_threshold:
                    self.drive(self.FAST, self.FAST)  # move forward
                else:
                    # Close enough → stop
                    self.status_code = 3
                    self.just_switched = True
        else:
            # Lost the ball → search again
            self.status_code = 1


    # GOAAAL
    def kick(self):
        """
        Stop MiRo when in front of the ball.
        """
        if self.just_switched:
            print("Miro reached the ball, stopping!")
            self.just_switched = False
        self.drive(0.0, 0.0)
        # Reset to search again if needed
        self.status_code = 0
        self.just_switched = True


    def update_heading_estimate(self, wl, wr):
        """
        Uses wheel speeds to estimate yaw rotation.
        This is not perfect but is sufficient for search navigation.
        """
        now = rospy.get_time()
        dt = now - self.last_time
        self.last_time = now

        # Convert wheel speeds (m/s) to angular velocity estimate (deg/s)
        wheel_base = 0.148  # MiRo wheel separation in meters
        omega = (wr - wl) / wheel_base  # rad/s
        self.current_heading += math.degrees(omega * dt)
        self.current_heading %= 360.0

    def __init__(self):
        # Initialise a new ROS node to communicate with MiRo, if needed
        if not self.IS_MIROCODE:
            rospy.init_node("kick_blue_ball", anonymous=True)
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
        # Create a new publisher to move the robot head
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
        # Remember which direction the ball is facing
        self.last_seen_side = None # 0 = Left, 1 = right

        self.obstacle_avoider = ObstacleAvoidance(topic_base_name)


        # Navigation memory
        self.visited_map = set()
        self.current_heading = 0.0  # deg
        self.last_time = rospy.get_time()

        # Avoid infinite spinning
        self.total_headings = 36  # 36 × 10° steps = full 360 sweep
        self.heading_resolution = 10  # degrees

    def loop(self):
        """
        Main control loop
        """
        print("MiRo plays ball, press CTRL+C to halt...")
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


# This condition fires when the script is called directly
if __name__ == "__main__":
    main = MiRoClient()  # Instantiate class
    main.loop()  # Run the main control loop