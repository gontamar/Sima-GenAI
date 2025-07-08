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

import base64
import http.server
import json
import os
import requests
import socketserver

PORT = 9998

# Utility class to listen for the incoming requests
class HttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Invalid JSON')
            return
        
        text = data.get('text', '')
        image = data.get('image', '')
        if image:
            image_bytes = base64.b64decode(image)
            image_path = os.path.join('/tmp/llava.jpg')
            with open(image_path, 'wb') as f:
                f.write(image_bytes)

        print(f'Got {text} and image from remote to run llava')
        if not text:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Missing "text" field in JSON')
            return

        processed_text = text

        # Response from here
        payload = {'text': processed_text}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True  # Enable SO_REUSEADDR

        
Handler = HttpRequestHandler

with ReusableTCPServer(("", PORT), Handler) as httpd:
    print(f"Serving on port {PORT}")
    httpd.serve_forever()
