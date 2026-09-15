#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sds_drone_generator import make_sds_drone_frame
from toda import estimate_angle_toda   # deine TODA-Funktion

def test_toda(angle, distance, noise):
    mic = make_sds_drone_frame(angle, distance, noise)
    est = estimate_angle_toda(mic)
    print(f"True={angle}°, TODA={est:.1f}°")

angles = [0,45,90,135,180,225,270,315]
for a in angles:
    test_toda(a, 2.0, 0.1)
