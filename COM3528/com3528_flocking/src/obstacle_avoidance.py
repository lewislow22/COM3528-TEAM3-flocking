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
    """
    
    This class handles real-time obstacle detection and avoidance using sonar,
    cliff, and light sensors. It's designed to work alongside navigation systems -
    you give it your desired wheel speeds, and it'll override them if it detects danger.
    
    NOTE: This worked really well when we merged it with navigation.py - acts as a 
    safety layer on top of the path planning, catching obstacles the main nav might miss.
    """
    
    def __init__(self, topic_root):
        
        # Set up publisher so we can send movement commands to the robot
        self.pub_cmd_vel = rospy.Publisher(
            topic_root + "/control/cmd_vel", TwistStamped, queue_size=0
        )
        self.vel = TwistStamped()

        # Store the latest sensor readings - start with safe default values
        self.sonar = 1.0  # Distance to obstacle 
        self.cliff_left = 1.0  # Cliff sensor values (1.0 = solid ground)
        self.cliff_right = 1.0  # We don't really need these since we're on flat ground, but doesn't hurt to have
        self.light_fl = 0.0  # Light sensor values (0.0 = dark, higher = brighter)
        self.light_fr = 0.0

        # Subscribe to all the sensor topics we care about
        rospy.Subscriber(topic_root + "/sensors/sonar", Range, self.sonar_cb)
        rospy.Subscriber(topic_root + "/sensors/cliff", UInt16, self.cliff_cb)
        rospy.Subscriber(topic_root + "/sensors/light", Float32MultiArray, self.light_cb)

        # Thresholds for when we need to react to things
        self.SONAR_MIN = 0.16  
        self.SONAR_FAR = 0.35  
        self.BACKUP_SPEED = -0.15  
        self.TURN_SPEED = 0.40  

        self.LIGHT_THRESHOLD = 0.75  # Anything brighter than this is worth avoiding
        self.CLIFF_THRESHOLD = 0.2  # Below this means we're at an edge - not really needed for our setup but good to have


    # Callback functions - these get called whenever new sensor data arrives
    def sonar_cb(self, msg):
        self.sonar = msg.range

    def cliff_cb(self, msg):
        try:
            # Cliff data comes as an array with left and right values
            self.cliff_left = msg.data[0]
            self.cliff_right = msg.data[1]
        except:
            pass  # If something goes wrong, just keep the old values

    def light_cb(self, msg):
        # Front-left and front-right light sensors
        self.light_fl = msg.data[0]
        self.light_fr = msg.data[1]


    def avoidance_required(self):
        """
        Checking if we need to do any avoiding right now.
        Returns whether we need to avoid, plus a list of reasons why.
        """
        reasons = []

        # Too close to something in front
        if self.sonar < self.SONAR_FAR:
            reasons.append("sonar")

        # About to drive off an edge (not really a concern on flat ground, but kept as a safety feature)
        if self.cliff_left < self.CLIFF_THRESHOLD or self.cliff_right < self.CLIFF_THRESHOLD:
            reasons.append("cliff")

        # Seeing something really bright (probably a wall or obstacle)
        if self.light_fl > self.LIGHT_THRESHOLD or self.light_fr > self.LIGHT_THRESHOLD:
            reasons.append("light_wall")

        return len(reasons) > 0, reasons


    def compute_avoidance(self):
        """
        Figure out what wheel speeds we need to avoid the obstacle.
        Returns (left_wheel_speed, right_wheel_speed).
        Priority order: cliffs first, then sonar, then light.
        """
        
        # PRIORITY 1: Don't fall off edges! (Unlikely to trigger on flat surfaces, but nice safety net)
        if self.cliff_left < self.CLIFF_THRESHOLD or self.cliff_right < self.CLIFF_THRESHOLD:
            # Back up and turn away from whichever side detected the cliff
            if self.cliff_left < self.cliff_right:
                return self.BACKUP_SPEED, -self.BACKUP_SPEED  # Turn right while backing up
            else:
                return -self.BACKUP_SPEED, self.BACKUP_SPEED  # Turn left while backing up

        # PRIORITY 2: Deal with obstacles detected by sonar
        if self.sonar < self.SONAR_FAR:
            
            # Way too close - just back straight up
            if self.sonar < self.SONAR_MIN:
                return self.BACKUP_SPEED, self.BACKUP_SPEED

            # Close enough to worry about - turn in place away from brighter side
            # (assumption: the obstacle is probably on the brighter side)
            if self.light_fl > self.light_fr:
                return 0.0, +0.20  # Turn right
            else:
                return +0.20, 0.0  # Turn left

        # PRIORITY 3: Avoid bright areas (walls, reflective surfaces, etc.)
        if self.light_fl > self.LIGHT_THRESHOLD or self.light_fr > self.LIGHT_THRESHOLD:
            # Turn away from the brighter side
            if self.light_fl > self.light_fr:
                return 0.0, +0.10  # Turn right gently
            else:
                return +0.10, 0.0  # Turn left gently

        # All clear - no avoidance needed
        return 0.0, 0.0


    def step(self, base_left, base_right):
        """
        Main control loop step. Give it the wheel speeds you WANT to use,
        and it'll either use them (if safe) or override with avoidance.
        
        When integrated with navigation.py, the nav system passes its computed wheel speeds
        here, and this acts as a safety override. Worked great in testing!
        
        Returns whether avoidance was triggered and why.
        """
        
        # Check if we need to avoid anything
        avoid_now, reasons = self.avoidance_required()

        if avoid_now:
            # Override the requested speeds with avoidance behavior
            left, right = self.compute_avoidance()
        else:
            # All clear - use the speeds that were requested
            left, right = base_left, base_right

        # Convert wheel speeds to the velocity command format ROS expects
        (dr, dtheta) = wheel_speed2cmd_vel([left, right])
        self.vel.twist.linear.x = dr  # Forward/backward speed
        self.vel.twist.angular.z = dtheta  # Turning speed
        
        # Send the command to the robot
        self.pub_cmd_vel.publish(self.vel)

        return avoid_now, reasons