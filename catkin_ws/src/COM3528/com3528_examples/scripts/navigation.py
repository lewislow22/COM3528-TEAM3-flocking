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
        Drives MiRo with obstacle avoidance.
        - We propose wheel speeds (speed_l, speed_r)
        - ObstacleAvoidance overrides them if needed
        - ObstacleAvoidance publishes the cmd_vel message
        """

        # Update rotated-heading estimate (your SLAM-lite behaviour)
        avoid, reasons = self.obstacle_avoider.step(speed_l, speed_r)
        if not avoid:
            self.update_heading_estimate(speed_l, speed_r)

        # Push desired wheel speeds into obstacle avoidance module
        # This returns (avoid_now, reasons)
        # and internally publishes the TwistStamped (cmd_vel)
        self.obstacle_avoider.step(speed_l, speed_r)

        # Nothing else to publish here — ObstacleAvoidance did it.


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
        Search using navigation:
        - Avoid revisiting previously searched headings
        - Avoid obstacles
        - Sweep through all headings systematically
        """
        if self.just_switched:
            print("MiRo beginning structured search…")
            self.just_switched = False

        # 1. Try to detect ball normally
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
            self.status_code = 2
            self.just_switched = True
            return

        # 2. Check for obstacle
        avoid, reasons = self.obstacle_avoider.avoidance_required()
        if avoid:
            print("Obstacle detected — avoiding:", reasons)
            self.drive(-self.SLOW, self.SLOW)
            return

        # 3. Quantize heading
        quant = int(self.current_heading // self.heading_resolution) * self.heading_resolution

        if quant not in self.visited_map:
            # New heading → mark visited and scan
            print(f"Exploring new heading {quant}°")
            self.visited_map.add(quant)
            self.drive(0.0, 0.0)  # pause briefly to scan
            return

        # 4. If heading already visited → rotate to next unvisited
        for i in range(1, self.total_headings + 1):
            next_heading = (quant + i * self.heading_resolution) % 360
            if next_heading not in self.visited_map:
                # Rotate toward next unvisited direction
                print(f"Rotating to next unvisited heading {next_heading}°")
                self.drive(self.SLOW, -self.SLOW)
                return

        # 5. If all headings visited → stop or reset
        print("Full 360° search completed — no ball found.")
        self.drive(0.0, 0.0)



    def lock_onto_ball(self, error=25):
        """
        [2 of 3] Once a ball has been detected, turn MiRo to face it
        """
        if self.just_switched:  # Print once
            print("MiRo is locking on to the ball")
            self.just_switched = False
        for index in range(2):  # For each camera (0 = left, 1 = right)
            # Skip if there's no new image, in case the network is choking
            if not self.new_frame[index]:
                continue
            image = self.input_camera[index]
            # Run the detect ball procedure
            self.ball[index] = self.detect_ball(image, index)
        # If only the right camera sees the ball, rotate clockwise
        if not self.ball[0] and self.ball[1]:
            self.drive(self.SLOW, -self.SLOW)
        # Conversely, rotate counter-clockwise
        elif self.ball[0] and not self.ball[1]:
            self.drive(-self.SLOW, self.SLOW)
        # Make the MiRo face the ball if it's visible with both cameras
        elif self.ball[0] and self.ball[1]:
            error = 0.05  # 5% of image width
            # Use the normalised values
            left_x = self.ball[0][0]  # should be in range [0.0, 0.5]
            right_x = self.ball[1][0]  # should be in range [-0.5, 0.0]
            rotation_speed = 0.03  # Turn even slower now
            if abs(left_x) - abs(right_x) > error:
                self.drive(rotation_speed, -rotation_speed)  # turn clockwise
            elif abs(left_x) - abs(right_x) < -error:
                self.drive(-rotation_speed, rotation_speed)  # turn counter-clockwise
            else:
                # Successfully turned to face the ball
                self.status_code = 3  # Switch to the third action
                self.just_switched = True
                self.bookmark = self.counter
        # Otherwise, the ball is lost :-(
        else:
            self.status_code = 0  # Go back to square 1...
            print("MiRo has lost the ball...")
            self.just_switched = True

    # GOAAAL
    def kick(self):
        "Once in position, Miro stops at the ball instead of kicking it"
        if self.just_switched:
            print("Miro reached the ball, stopping!")
            self.just_switched = False
        self.drive(0.0, 0.0)
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