# High-Level Flow and Architecture of `gontamar/Sima-GenAI`

## 1. Overview
This repository is a demo package to run the SiMa.Ai GenAI demo frontend. Its primary function is to provide a web interface that interacts with the SiMa.ai modalix board, enabling users to test GenAI features such as image/audio capture, voice recording, and display AI-generated results.

---

## 2. Main Components

### a. `app.py` (Python Webserver)
- **Flask-based backend**: Hosts a web server using Flask, enhanced with Flask-SocketIO for real-time communication between backend and frontend.
- **Routes**:
  - `/` - Renders the main web interface (`index.html`).
  - `/capture_and_send` - Captures an image from the camera, encodes it, and sends it to the frontend.
  - `/video_feed` - Streams live video from the camera.
  - `/upload` - Handles file uploads (audio/image); processes them using AI models and returns results.
- **Backend logic**:
  - Integrates with AI models (for transcription, image analysis, etc.).
  - Uses classes like `ModelManager`, `TalkController`, `AppContext` to manage models, speech, and application state.
  - Handles uploading, saving, and processing media files, returning results as needed.

### b. `templates/index.html` (Frontend HTML)
- The main user interface for the demo.
- Displays:
  - Live camera stream and image preview.
  - Results box for AI-generated responses.
  - Controls for recording audio, capturing images, analyzing scenes, and resetting the interface.
- Loads required CSS/JS for styling and functionality.

### c. `static/main.js` (Frontend JS)
- Handles UI interactions:
  - Manages recording audio, capturing images, and sending them to the backend.
  - Receives responses (text/audio) from the backend via web sockets and updates the UI.
  - Handles speech synthesis for AI-generated text.
  - Shows results in a streaming/typing animation.
- Responds to various user actions (button clicks, etc.).

### d. `static/style.css`
- Styles the interface, including the results box, question display, controls, and layout.

### e. `certs/`
- Contains SSL certificates for HTTPS, which are required for secure media uploads/downloads.

### f. `test/`
- Contains testing code (not detailed in the snippets above).

---

## 3. Flow Diagram (Textual)

```
User (Browser)
  |
  |--(loads)--> index.html (UI)
  |                |
  |                |--(clicks/records/uploads)---+
  |--(HTTP/WebSocket)----------------------------+--> Flask app (app.py)
                                                  |
                                                  |--(routes requests)
                                                  |--(manages image/audio uploads)
                                                  |--(invokes AI/ML models)
                                                  |--(returns results)
                                                  |
  |<--(WebSocket/HTTP)-- results/updates ---------+
  |                |
  |--(UI updates/results/audio playback)----------+
```

---

## 4. Key Interactions

- **User records audio or uploads image** via the web UI.
- **Frontend JS** sends data to the backend using HTTP POST or WebSocket.
- **Backend** processes the data using AI models, e.g., transcribes speech, analyzes scene, etc.
- **Backend** sends results (text, audio, images) back to the frontend.
- **Frontend JS** updates the UI, displays results, and (optionally) uses browser speech synthesis to "talk" results.

---

## 5. How the Main Modules Interact

- **Frontend (JS/HTML/CSS)**:
  - Handles user input, displays results, and communicates with backend via Socket.IO and HTTP.
- **Backend (Flask/Python)**:
  - Serves the UI, manages real-time communication, processes media with AI models, and maintains application state.
- **AI Model Layer**:
  - Encapsulated in `ModelManager`, called by Flask endpoints to process uploads and generate results.
- **Media and Streaming**:
  - Video stream handled by Flask route (`/video_feed`), image capture by `/capture_and_send`.
  - Audio and image uploads handled by `/upload`.

---

## 6. Entry Points

- **Start server**: `python3 app.py` (with camera and board IP parameters).
- **Access UI**: Open `https://127.0.0.1:5000/` in Google Chrome.

---

## 7. Summary

- The repo is structured as a Python Flask app serving a real-time AI demo web interface.
- The frontend interacts with the backend using HTTP and WebSocket, enabling live camera, audio, and AI-powered interaction.
- AI model logic is abstracted behind classes and is triggered automatically by user actions in the browser.

---

For further details, see the [README](https://github.com/gontamar/Sima-GenAI/blob/main/README.md) or browse the code for `app.py` and `static/main.js`.
