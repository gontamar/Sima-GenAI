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

from flask import Flask, Response, render_template, jsonify, request
import cv2
import requests
import io

app = Flask(__name__)

# URL of the server where the captured image will be sent
DESTINATION_SERVER_URL = 'http://127.0.0.1:8000/upload'

# Initialize the camera
camera = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

def generate_video_stream():
    """Video streaming generator function."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def capture_image():
    """Capture a single frame from the camera."""
    success, frame = camera.read()
    if not success:
        print("Error: Could not capture frame.")
        return None
    _, jpeg_image = cv2.imencode('.jpg', frame)
    return jpeg_image.tobytes()

def send_image_to_server(image_data):
    """Send the captured image to the destination server."""
    files = {'file': ('image.jpg', io.BytesIO(image_data), 'image/jpeg')}
    response = requests.post(DESTINATION_SERVER_URL, files=files)
    return response.status_code, response.text

@app.route('/')
def index():
    """Home page with video stream and capture button."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for video streaming."""
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_and_send', methods=['POST'])
def capture_and_send():
    """Capture an image and send to the destination server."""
    image_data = capture_image()
    if image_data is None:
        return jsonify({'error': 'Failed to capture image'}), 500
    status_code, response_text = send_image_to_server(image_data)
    return jsonify({'status_code': status_code, 'response': response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
