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
import argparse
import base64
import cv2
import http.server
import json
import logging
import os
import pyttsx3
import requests
import shutil
import socketserver
import re
import sys
import threading

import whisper
from queue import Queue

# Flask imports
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

camera = None
genai_app = None

class AppConstants:
    DEFAULT_SIMA_SERVER_IP = "192.168.1.20:9998"
    DEFAULT_CAMERA_IDX = 0
    DEFAULT_LLAVA_QUERY_STR='Describe what you see in the picture.'
    DEFAULT_HTTP_PORT = 8081
    DEFAULT_CAMERA_IDX = 0
    DEFAULT_UPLOADS_DIR = 'uploads'
       
class ModelManager:
    def __init__(self):
        self.model = None

    def load(self):
        self.model = whisper.load_model("small")

    def run(self, path) -> str:
        return self.model.transcribe(path, language="en")

class TalkController:
    def __init__(self):
        self._next = None
        self.prefix = ''
        self.totalk = ''
        self.talk = []
        
    def update(self, subword):
        mod_subword = ''

        if (subword != 'END'):  # End of streaming
            # Compensate for newline unicode
            if ('<0x0A>' in subword):
                mod_subword = re.sub(r"<0x([0-9A-Fa-f]+)>", "", subword)
            
            if ('</s>' in subword):
                mod_subword = re.sub(r"</s>", "", subword)
            
            if (self.check_punctuation(subword)):
                # Kludge
                if mod_subword != '':
                    self.prefix_s = mod_subword.split('.')
                    tmp = self.prefix_s[0] + '.'
                    self._next = self.prefix_s[-1]
                    self.talk.append(tmp)
                else:
                    self.talk.append(subword)
                self.totalk = self.generate_talk()
                logging.info(f'after talking {self.talk}, {self.totalk}')
                genai_app.emit('talk', {"results": self.totalk.strip()})
                self.talk = []
                if self._next is not None:
                    self.talk.append(self._next)
                    self._next = None
            else:
                self.talk.append(subword)

    def reset(self):
        self._next = None
        self.prefix = ''
        self.totalk = ''

    def check_punctuation(self, word):
        return bool(re.search(r"[\.]", word))
        
    def generate_talk(self):
        return " ".join(self.talk)

class AppContext:
    def __init__(self):
        self.app = None
        self.model_manager = ModelManager()
        self.uploaded_video_filename = None
        self.talk_ctrl = TalkController()
        self.socketio = None
        
    def update_settings(self, camidx, llava_server_ip):
        self.camidx = AppConstants.DEFAULT_CAMERA_IDX if camidx is None else camidx
        self.llava_server_ip = AppConstants.DEFAULT_SIMA_SERVER_IP if llava_server_ip is None else llava_server_ip
        self.update_config()

    def update_config(self):
        self.app.config['CAMERA_IDX'] = self.camidx
        self.app.config['SIMAAI_IP_ADDR'] = self.llava_server_ip
        self.app.config['SIMAAI_IP_PORT'] =  AppConstants.DEFAULT_HTTP_PORT
        self.app.config['UPLOAD_FOLDER'] = AppConstants.DEFAULT_UPLOADS_DIR

    def get_config(self):
        return self.app.config
        
    def initialize(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app)
        
        if not os.path.exists(AppConstants.DEFAULT_UPLOADS_DIR):
            os.makedirs(AppConstants.DEFAULT_UPLOADS_DIR)

        self.model_manager.load()
        self.setup_router()

    def emit(self, ep, obj):
        self.socketio.emit(ep, obj)

    def run(self):
        self.socketio.run(self.app, host='0.0.0.0', port="5000",
                          ssl_context=('certs/server.crt', 'certs/server.key'),
                          debug=False, allow_unsafe_werkzeug=True)

    def setup_router(self):
        @self.app.route('/capture_and_send', methods=['POST'])
        def capture_and_send():
            """Capture an image, send it to the destination server, and return it as Base64."""
            image_data = capture_image()
            if image_data is None:
                return jsonify({'error': 'Failed to capture image'}), 500

            # Encode the image data in Base64 to display on the page
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            image_src = f"data:image/jpeg;base64,{encoded_image}"
            return jsonify({'status_code': 200, 'response': 'Done', 'image_src': image_src})

        @self.app.route('/video_feed')
        def video_feed():
            """Route for video streaming."""
            return Response(generate_video_stream(self.camidx),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/')
        def index():
            self.socketio.emit('update', {"hello" : "world"})
            return render_template('index.html')


        @self.app.route('/upload', methods=['POST'])
        def upload():
            audio_file = None
            image_file = None
            query_str = AppConstants.DEFAULT_LLAVA_QUERY_STR
    
            if 'audio_data' in request.files:
                audio_file = request.files['audio_data']
            if 'image_data' in request.files:
                image_file = request.files['image_data']

            cfg = genai_app.get_config()
    
            if audio_file:
                audio_file.save(os.path.join(cfg['UPLOAD_FOLDER'], audio_file.filename))

            image_path = None
            if image_file:
                image_path = os.path.join(cfg['UPLOAD_FOLDER'], image_file.filename)
                image_file.save(image_path)
        
            if audio_file:
                result = genai_app.model_manager.run("uploads/audio.webm")
                logging.info(f"Transcribed results {result['text']}")
                query_str = result["text"]
        
            if ('Thank you' in query_str) or ('Thanks' in query_str):
                query_str = AppConstants.DEFAULT_LLAVA_QUERY_STR

            logging.info(f"Query string {query_str}")
            thread = threading.Thread(target=post_to_sima, args=[query_str, image_path])
            thread.start()
            return {'question' : query_str}

        @self.app.route('/upload_image', methods=['POST'])
        def upload_image():
            image_file = None
            query_str = AppConstants.DEFAULT_LLAVA_QUERY_STR

            if 'image_data' in request.files:
                image_file = request.files['image_data']

            cfg = genai_app.get_config()
                
            image_path = None
            if image_file:
                image_path = os.path.join(cfg['UPLOAD_FOLDER'], image_file.filename)
                image_file.save(image_path)
        
            logging.info(f"Query string {query_str}")
            thread = threading.Thread(target=post_to_sima, args=[query_str, image_path])
            thread.start()
            return {'question' : query_str}
        
class HttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        """Handle POST request."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(post_data)
            response = {"status": "Received data successfully"}
            self.send_response(200)
        except json.JSONDecodeError:
            logging.error("Invalid JSON received in standalone HTTP Server")
            response = {"status": "Invalid JSON"}
            self.send_response(400)

        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))
        send_talk_text(data['text'])
        genai_app.emit('update', {"results": data['text']})

def send_talk_text(in_text):
    talk = genai_app.talk_ctrl.update(in_text)
            
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True
        
def start_http_server():
    """Start the standalone HTTP server in a separate thread."""
    cfg = genai_app.get_config()

    with ReusableTCPServer(("", cfg['SIMAAI_IP_PORT']), HttpRequestHandler) as httpd:
        logging.info(f"Standalone HTTP server is running on port {cfg['SIMAAI_IP_PORT']}")
        httpd.serve_forever()

def cleanup_data():
    logging.info('Cleaning up all cached images and audio files')
    if os.path.exists('./uploads/audio.webm'):
        os.remove('./uploads/audio.webm')

    if os.path.exists('./uploads/camera.jpg'):
        os.remove('./uploads/camera.jpg')

    if os.path.exists('./uploads/image.jpg'):
        os.remove('./uploads/image.jpg')
        
def cleanup():
    shutil.rmtree('./uploads')
    os.mkdir('./uploads')

# Function to post the file to another server
def post_to_sima(text, image_path = None):
    genai_app.emit('update', {"progress" : "SiMa.ai is processing, please wait.."})
    cfg = genai_app.get_config()
    url =  'http://' + str(cfg['SIMAAI_IP_ADDR']).strip()
    base64_image = None

    if image_path is None:
        logging.debug(f'Searching for camera image if captured')
        if os.path.exists('./uploads/camera.jpg'):
            logging.debug(f'Camera image available using the same with query')
            image_path = './uploads/camera.jpg'

    if image_path is not None:
        with open(image_path, 'rb') as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
    logging.debug(f'Posting to sima llama server running {url}')
    response = requests.post(url, json={'text': text, 'image' : base64_image})
    logging.debug(response.json())
    cleanup_data()
    
def generate_video_stream(source):
    """Video streaming generator function."""
    global camera
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        raise IOError(f'Cannot open video source {source}')
            
    while True:
        # Capture frame-by-frame
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
    cv2.imwrite('./uploads/camera.jpg', frame)
    print("Captured image from camera and wrote to uploads/camera.jpg")
    return jpeg_image.tobytes()

if __name__ == '__main__':
    log_filename = 'server.log'
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        encoding='utf-8',
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    genai_app = AppContext()
    genai_app.initialize()
    
    parser = argparse.ArgumentParser(description ='LTTS MMAI demo application args')
    parser.add_argument('--camidx', type=int, required=False)
    parser.add_argument('--ip', type=str, required=False)
    args = parser.parse_args()

    genai_app.update_settings(args.camidx, args.ip)
    cleanup()

    logging.info(f'Starting SiMa.ai genai server with {args.camidx} {args.ip}')
    server_thread = threading.Thread(target=start_http_server)
    server_thread.daemon = True
    server_thread.start()
    logging.info("Started standalone HTTP server in a separate thread")
    genai_app.run()
