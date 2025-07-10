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
import json
import logging
import os
import requests # Still needed if you want to keep the option to connect to a real server later
import shutil
import re
import sys
import threading
import time # Added for simulating delay

import whisper
from queue import Queue

# Flask imports
from flask import Flask, Response, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

camera = None
genai_app = None

class AppConstants:
    # Changed default IP to a placeholder as we're mocking
    DEFAULT_SIMA_SERVER_IP = "mock_sima_server:9998"
    DEFAULT_CAMERA_IDX = 0
    DEFAULT_LLAVA_QUERY_STR='Describe what you see in the picture.'
    # DEFAULT_HTTP_PORT was for the incoming server, which we'll remove
    DEFAULT_UPLOADS_DIR = 'uploads'
       
class ModelManager:
    def __init__(self):
        self.model = None

    def load(self):
        # Using a smaller model for faster local download/inference
        # You can change "small" to "base" or "medium" if you prefer
        logging.info("Loading Whisper model 'small'...")
        self.model = whisper.load_model("small")
        logging.info("Whisper model loaded.")

    def run(self, path) -> str:
        logging.info(f"Transcribing audio from {path}...")
        result = self.model.transcribe(path, language="en")
        logging.info(f"Transcription complete: {result['text']}")
        return result

class TalkController:
    def __init__(self):
        self._next = None
        self.prefix = ''
        self.totalk = ''
        self.talk = []
        
    def update(self, subword):
        mod_subword = ''

        if (subword == 'END'):  # End of streaming
            if self.talk: # If there's any remaining text not ended by punctuation
                self.totalk = self.generate_talk()
                logging.info(f'after talking (END of stream) {self.talk}, {self.totalk}')
                genai_app.emit('talk', {"results": self.totalk.strip()})
            self.reset() # Reset for next session
            return

        # Compensate for newline unicode
        if ('<0x0A>' in subword):
            mod_subword = re.sub(r"<0x([0-9A-Fa-f]+)>", "", subword)
        else:
            mod_subword = subword # Use original if no newline char

        if ('</s>' in mod_subword):
            mod_subword = re.sub(r"</s>", "", mod_subword)
        
        # Check for punctuation at the end of the modified subword
        if (self.check_punctuation(mod_subword)):
            # Kludge: If punctuation found, treat it as end of a sentence
            if mod_subword != '':
                # Attempt to split by period and handle what comes after
                parts = mod_subword.split('.')
                if len(parts) > 1:
                    # The first part plus the period is the complete sentence
                    sentence_to_add = parts[0].strip() + '.'
                    self.talk.append(sentence_to_add)
                    # The rest (if any) becomes the start of the next sentence
                    self._next = '.'.join(parts[1:]).strip()
                else:
                    # If no split happened, just add the word
                    self.talk.append(mod_subword.strip())
            else:
                self.talk.append(subword.strip()) # Fallback if mod_subword somehow empty
            
            self.totalk = self.generate_talk()
            logging.info(f'after talking (punctuation) {self.talk}, {self.totalk}')
            genai_app.emit('talk', {"results": self.totalk.strip()})
            self.talk = [] # Clear for the next sentence
            if self._next: # If there was a remainder, add it
                self.talk.append(self._next)
                self._next = None
        else:
            self.talk.append(mod_subword.strip()) # Add non-punctuated word directly

    def reset(self):
        logging.info("TalkController reset.")
        self._next = None
        self.prefix = ''
        self.totalk = ''
        self.talk = []

    def check_punctuation(self, word):
        # Look for period, question mark, or exclamation mark at the end of the word
        return bool(re.search(r"[\.?!]$", word.strip()))
        
    def generate_talk(self):
        return " ".join(self.talk)

class AppContext:
    def __init__(self):
        self.app = None
        self.model_manager = ModelManager()
        self.talk_ctrl = TalkController()
        self.socketio = None
        
    def update_settings(self, camidx, llava_server_ip):
        self.camidx = AppConstants.DEFAULT_CAMERA_IDX if camidx is None else camidx
        self.llava_server_ip = AppConstants.DEFAULT_SIMA_SERVER_IP if llava_server_ip is None else llava_server_ip
        self.update_config()

    def update_config(self):
        self.app.config['CAMERA_IDX'] = self.camidx
        self.app.config['SIMAAI_IP_ADDR'] = self.llava_server_ip # Will be our mock placeholder
        # self.app.config['SIMAAI_IP_PORT'] = AppConstants.DEFAULT_HTTP_PORT # Not needed for mock
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
        # Removed ssl_context for easier local development
        self.socketio.run(self.app, host='0.0.0.0', port=5000,
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
            # Emit an initial message on connection (client-side Socket.IO handles connection)
            # This 'update' emit might be better placed in an on_connect handler if truly needed on every client connection
            # self.socketio.emit('update', {"hello" : "world"}) 
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
                audio_path = os.path.join(cfg['UPLOAD_FOLDER'], 'audio.webm')
                audio_file.save(audio_path)
                logging.info(f"Audio file saved to {audio_path}")

            image_path = None
            if image_file:
                image_path = os.path.join(cfg['UPLOAD_FOLDER'], 'image.jpg') # Standardize name for simplicity
                image_file.save(image_path)
                logging.info(f"Image file saved to {image_path}")
        
            if audio_file:
                # Use the saved audio file path for whisper
                result = genai_app.model_manager.run(audio_path)
                query_str = result["text"]
        
            # Handle specific queries or provide a default
            if ('Thank you' in query_str) or ('Thanks' in query_str):
                query_str = "You are welcome! What else can I help you with?"
                self.talk_ctrl.reset() # Reset talk controller for a new conversation
            elif not query_str.strip(): # If transcription results in empty string
                 query_str = AppConstants.DEFAULT_LLAVA_QUERY_STR # Fallback if voice input is empty

            logging.info(f"Final Query string: '{query_str}'")
            # Call the modified post_to_sima directly, no external request
            thread = threading.Thread(target=post_to_sima, args=[query_str, image_path])
            thread.start()
            return jsonify({'question' : query_str})

        @self.app.route('/upload_image', methods=['POST'])
        def upload_image():
            image_file = None
            query_str = AppConstants.DEFAULT_LLAVA_QUERY_STR

            if 'image_data' in request.files:
                image_file = request.files['image_data']

            cfg = genai_app.get_config()
                
            image_path = None
            if image_file:
                image_path = os.path.join(cfg['UPLOAD_FOLDER'], 'image.jpg') # Standardize name
                image_file.save(image_path)
                logging.info(f"Uploaded image saved to {image_path}")
        
            logging.info(f"Query string for image: '{query_str}'")
            thread = threading.Thread(target=post_to_sima, args=[query_str, image_path])
            thread.start()
            return jsonify({'question' : query_str})
        
# Removed HttpRequestHandler, ReusableTCPServer, and start_http_server
# as we are mocking the SiMa.ai server response directly within post_to_sima.

def cleanup_data():
    logging.info('Cleaning up all cached images and audio files')
    if os.path.exists('./uploads/audio.webm'):
        os.remove('./uploads/audio.webm')

    if os.path.exists('./uploads/camera.jpg'):
        os.remove('./uploads/camera.jpg')

    if os.path.exists('./uploads/image.jpg'):
        os.remove('./uploads/image.jpg')
        
def cleanup():
    logging.info('Performing full cleanup of uploads directory.')
    if os.path.exists('./uploads'):
        shutil.rmtree('./uploads')
    os.makedirs('./uploads')
    logging.info('Uploads directory recreated.')

# Function to simulate posting to Sima and receiving a response
def post_to_sima(text, image_path = None):
    genai_app.emit('update', {"progress" : "Processing request, please wait..."})
    logging.info(f"Simulating processing for query: '{text}' and image_path: '{image_path}'")

    # Simulate network latency and processing time
    time.sleep(2) 
    
    # --- Mock SiMa.ai Response Logic ---
    mock_response_text = ""
    if "hello" in text.lower() or "hi" in text.lower():
        mock_response_text = "Hello there! I am a large language model. How can I assist you today?"
    elif "time" in text.lower():
        mock_response_text = f"The current time is {time.strftime('%H:%M:%S')}."
    elif "date" in text.lower():
        mock_response_text = f"Today's date is {time.strftime('%Y-%m-%d')}."
    elif "weather" in text.lower():
        mock_response_text = "I cannot access real-time weather information, but I hope it's a lovely day wherever you are!"
    elif "describe what you see" in text.lower() or "what is in the picture" in text.lower():
        if image_path and os.path.exists(image_path):
            mock_response_text = "Based on the image, I see a typical indoor scene. It appears to be a desk with a monitor, possibly a keyboard, and some office supplies. There might be some personal items around too. It's a clear and well-lit photo."
        else:
            mock_response_text = "I was asked to describe the picture, but no image was provided or found. Please ensure an image is uploaded or captured for analysis."
    elif "thank you" in text.lower() or "thanks" in text.lower():
        mock_response_text = "You are most welcome! Is there anything else I can help you with?"
        genai_app.talk_ctrl.reset() # Reset conversation state
    else:
        mock_response_text = f"I received your request: '{text}'. This is a mock response. For a real interaction, please connect to a SiMa.ai device. I am designed to assist with various queries. Perhaps you could ask me about my capabilities or another question about the image."
    # --- End Mock Response Logic ---

    logging.info(f"Mock SiMa.ai response: '{mock_response_text}'")
    
    # Simulate streaming output
    # Split the response into words and send them one by one to simulate streaming
    words = mock_response_text.split()
    for i, word in enumerate(words):
        genai_app.talk_ctrl.update(word + (' ' if i < len(words) - 1 else '')) # Add space except for the last word
        time.sleep(0.1) # Small delay to make it seem like streaming

    genai_app.talk_ctrl.update('END') # Signal end of streaming
    genai_app.emit('update', {"progress" : "Response complete."}) # Final progress update
    cleanup_data() # Clean up temporary files

def generate_video_stream(source):
    """Video streaming generator function."""
    global camera
    # Ensure camera is initialized only once or re-initialized if closed
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(source)
        if not camera.isOpened():
            logging.error(f'Cannot open video source {source}. Make sure a webcam is connected.')
            # Yield an error frame or handle gracefully
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'Error: Camera not available' + b'\r\n')
            return
    
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()
        if not success:
            logging.warning("Failed to read frame from camera. Is it still available?")
            # Optionally try to re-open or break
            # camera.release()
            # camera = cv2.VideoCapture(source) # Attempt to re-open
            # if not camera.isOpened():
            break # Exit loop if camera fails permanently
        else:
            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def capture_image():
    """Capture a single frame from the camera."""
    global camera
    # Ensure camera is open before attempting to read
    if camera is None or not camera.isOpened():
        logging.error("Camera not open, cannot capture image.")
        return None
        
    success, frame = camera.read()
    if not success:
        logging.error("Error: Could not capture frame.")
        return None
    
    # Create the uploads directory if it doesn't exist
    if not os.path.exists(AppConstants.DEFAULT_UPLOADS_DIR):
        os.makedirs(AppConstants.DEFAULT_UPLOADS_DIR)

    image_path = os.path.join(AppConstants.DEFAULT_UPLOADS_DIR, 'camera.jpg')
    cv2.imwrite(image_path, frame)
    logging.info(f"Captured image from camera and wrote to {image_path}")
    
    _, jpeg_image = cv2.imencode('.jpg', frame) # Re-encode for in-memory use if needed
    return jpeg_image.tobytes()

if __name__ == '__main__':
    log_filename = 'server.log'
    logging.basicConfig(
        handlers=[
            logging.FileHandler(filename=log_filename, mode="w", encoding='utf-8'),
            logging.StreamHandler(sys.stdout) # Also log to console
        ],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    genai_app = AppContext()
    
    parser = argparse.ArgumentParser(description ='LTTS MMAI demo application args')
    parser.add_argument('--camidx', type=int, default=AppConstants.DEFAULT_CAMERA_IDX,
                        help=f'Camera index to use (default: {AppConstants.DEFAULT_CAMERA_IDX})')
    parser.add_argument('--ip', type=str, default=AppConstants.DEFAULT_SIMA_SERVER_IP,
                        help=f'SiMa.ai server IP and port (e.g., 192.168.1.20:9998). Default: {AppConstants.DEFAULT_SIMA_SERVER_IP} (mock)')
    args = parser.parse_args()

    genai_app.update_settings(args.camidx, args.ip)
    cleanup() # Clean up any prior run's uploads
    genai_app.initialize() # Initialize Flask app and load Whisper model

    logging.info(f'Starting SiMa.ai genai server with Camera Index: {args.camidx}, Mock SiMa IP: {args.ip}')
    # Removed the separate HTTP server thread as its functionality is now mocked in post_to_sima
    genai_app.run()
