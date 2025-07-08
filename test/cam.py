#########################################################
# Copyright (C) 2024 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################

import cv2

camera = cv2.VideoCapture(0)
def generate_video_stream():
    """Video streaming generator function."""
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()
        print(f'{frame.shape}')
        if not success:
            print('Breaking')
            break

if __name__ == "__main__":
    generate_video_stream()
