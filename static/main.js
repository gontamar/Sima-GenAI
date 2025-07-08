let mediaRecorder;
let audioChunks = [];

var intro = "Hello! Welcome to SiMa.ai Llama demo";


const socket = io();
const messageQueue = [];  // Queue to hold incoming messages
var talkQueue = [];

var results = document.getElementById("results");
var question = document.getElementById("question");
var reset = document.getElementById("reset");

let isProcessing = false;
let isTalking = false;
let isTyping = false;  // Flag to track if typing is in progress

window.onload = function (){
    console.log('Loading app');
}

function stopRecording() {
    mediaRecorder.stop();
    console.log("Recording stopped...");
}

function enableButtons() {
    document.getElementById("startRecord").disabled = false;
    document.getElementById("stopRecord").disabled = true;
    document.getElementById("capture").disabled = false;
    document.getElementById("sceneAnalyze").disabled = false;
    document.getElementById("reset").disabled = false;
}

function previewImage(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById('imagePreview');
            preview.src = e.target.result;
            preview.style.display = 'block';  // Show the image preview
        }
        reader.readAsDataURL(file);
    }
}

async function captureImage() {
    try {
        const response = await fetch('/capture_and_send', { method: 'POST' });
        const result = await response.json();

        // Update status and display the captured image
        // document.getElementById("status").textContent = "Image sent. Server response: " + result.response;
        
        // Display the captured image if available
        if (result.image_src) {
            const capturedImage = document.getElementById("imagePreview");
            capturedImage.src = result.image_src;
            capturedImage.style.display = "block";

            const fileInput = document.getElementById('fileInput');
            console.log(fileInput.files.length);
            fileInput.value = "";
        }
    } catch (error) {
        console.log("Error sending image" ,error);
    }
}

async function startRecording() {
    question.textContent = '';
    results.textContent = '';

    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length == 0)
        fileInput.value = '';
    
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = function(event) {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async function() {
        const formData = new FormData();

        // Check if a file is selected
        if (fileInput.files.length != 0) {
            formData.append('image_data', fileInput.files[0], 'image.jpg');
        }
        
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        audioChunks = [];

        formData.append('audio_data', audioBlob, 'audio.webm');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const jsonResponse = await response.json();
            question.style.visibility = 'visible';
            question.textContent = "Q: " + jsonResponse['question'];
        } catch (error) {
            console.error('Fetch error:', error);
        }
    };

    mediaRecorder.start();
    console.log("Recording started...");
}

async function sendQuery() {
    question.textContent = '';
    results.textContent = '';

    const formData = new FormData();

    // Check if a file is selected
    if (fileInput.files.length != 0) {
        formData.append('image_data', fileInput.files[0], 'image.jpg');
    }
    
    try {
        const response = await fetch('/upload_image', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const jsonResponse = await response.json();
        question.style.visibility = 'visible';
        question.textContent = "Q: " + jsonResponse['question'];
    } catch (error) {
        console.error('Fetch error:', error);
    }

    console.log("Recording started...");
}

socket.on('update', (data) => {
    if (data["results"]) {
        if (data["results"] == "END") {
            enableButtons();
            isTyping = false;  // Reset typing flag
            return;
        }
        messageQueue.push(data["results"]);  // Queue up incoming data
        processQueue();  // Process the queue
    } else {
        console.log("Unhandled progress item");
    }
});

socket.on('talk', (data) => {
    talkQueue.push(data["results"]);
    playTextAsSpeech();
});

// async function talk(text) {
//     var utterance = null;

//     console.log('Talking', text);
//     utterance = new SpeechSynthesisUtterance(w);
//     utterance.rate = 0.9;

//     window.speechSynthesis.speak(utterance);
//     utterance.onend = () => {

//     };
// }

async function processQueue() {
    if (isProcessing || messageQueue.length == 0) {
        console.log('Returning ', isProcessing);
        return;
    }

    isProcessing = true;
    
    if (isTyping || messageQueue.length === 0)
        return;

    var resp_text = messageQueue.shift();
    if (resp_text.includes('<0x0A>'))
        resp_text = resp_text.replace(/<0x0A>/g, '\n');

    if (resp_text.includes('</s>'))
        resp_text = resp_text.replace(/<\/s>/g, '');
    
    resp_text +=  " ";
    isTyping = true;
    
    if (resp_text === "END") {
        console.log('Done with processing, playing', results.textContent);
        enableButtons();
        isTyping = false;
        return;
    }
    
    results.style.visibility = 'visible';
    if (results.textContent.length === 0) {
        results.textContent = "MMAI says: ";
    }

    let index = 0;
    function typeCharacter() {
        if (index < resp_text.length) {
            results.textContent += resp_text.charAt(index);
            index++;
            setTimeout(typeCharacter, 5);
        } else {
            isTyping = false;
            processQueue();
            results.textContent += " ";
        }
    }

    typeCharacter();
    isProcessing = false;
}

function playTextAsSpeech() {
    // const textChunks = text.split(/(?<=[.!?])\s+/); // Split text by sentence endings
    if (isTalking || talkQueue.length == 0) {
        return;
    }

    isTalking = true;

    var textChunk = talkQueue.shift();
    console.log('Text', textChunk);
    
    const utterance = new SpeechSynthesisUtterance(textChunk.trim());
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);

    isTalking = false;
}

function resetImage() {
    const previewImage = document.getElementById("imagePreview");
    previewImage.src = '/static/default.jpg';

    const fileIn = document.getElementById('fileInput');
    fileIn.value = '';

    question.textContent = '';
    results.textContent = '';
}

document.getElementById("startRecord").addEventListener("click", function() {
    startRecording();
    document.getElementById("startRecord").disabled = true;
    document.getElementById("stopRecord").disabled = false;
    document.getElementById("capture").disabled = true;
    document.getElementById("sceneAnalyze").disabled = true;
    document.getElementById("reset").disabled = true;
});

document.getElementById("reset").addEventListener("click", function() {
    resetImage();
});

document.getElementById("sceneAnalyze").addEventListener("click", function() {
    sendQuery();
    document.getElementById("startRecord").disabled = true;
    document.getElementById("stopRecord").disabled = true;
    document.getElementById("capture").disabled = true;
    document.getElementById("sceneAnalyze").disabled = true;
    document.getElementById("reset").disabled = true;
});

document.getElementById("stopRecord").addEventListener("click", function() {
    stopRecording();
    document.getElementById("startRecord").disabled = true;
    document.getElementById("stopRecord").disabled = true;
    document.getElementById("capture").disabled = true;
});
