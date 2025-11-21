#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import miro_ros_interface as mri

miro_pub = mri.MiRoPublishers()
miro_pub.pub_tone(frequency=300, volume=20, duration=50)