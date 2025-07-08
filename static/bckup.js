let mediaRecorder;
let audioChunks = [];

// Start recording when record button is clicked
document.getElementById('recordButton').addEventListener('click', async () => {
    // Get audio stream from the user's device
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();

    // Change button colors
    document.getElementById('recordButton').style.backgroundColor = 'gray';
    document.getElementById('stopButton').style.backgroundColor = 'red';
    document.getElementById('recordButton').disabled = true;
    document.getElementById('stopButton').disabled = false;

    // Store the audio data in chunks
    audioChunks = [];
    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };
});

// Stop recording when stop button is clicked
document.getElementById('stopButton').addEventListener('click', () => {
    mediaRecorder.stop();

    // Change button colors back
    document.getElementById('recordButton').style.backgroundColor = 'green';
    document.getElementById('stopButton').style.backgroundColor = 'gray';
    document.getElementById('recordButton').disabled = false;
    document.getElementById('stopButton').disabled = true;

    // When recording stops, process the audio data and send to the server
    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/ogg' });
        const formData = new FormData();
        formData.append('audio_data', audioBlob, 'audio.ogg');

        fetch('/upload', {
            method: 'POST',
            body: formData
        }).then(response => {
            if (response.ok) {
                const jsonResp = await response.json();
                const simaText = jsonResp.processed_text;

                console.log("Sima response", simaText);
                playTextAsSpeech(simaText);        
            } else {
                alert('Failed to upload file.');
            }
        });
    };
});

// Function to convert text to speech and play it
function playTextAsSpeech(text) {
    // Check if SpeechSynthesis is supported
    if ('speechSynthesis' in window) {
        const speech = new SpeechSynthesisUtterance(text); // Create a new utterance
        window.speechSynthesis.speak(speech);             // Speak the text
    } else {
        console.error("Speech Synthesis not supported");
    }
}

