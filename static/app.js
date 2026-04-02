let mediaRecorder;
let audioChunks = [];
let isRecording = false;

const sentences = ["안녕하세요", "학교", "고려인", "사과", "바다", "대한민국"];
let currentSentenceIdx = 0;

function changeSentence() {
    currentSentenceIdx = (currentSentenceIdx + 1) % sentences.length;
    document.getElementById('target-sentence').innerText = sentences[currentSentenceIdx];
    document.getElementById('result-wrapper').classList.add('hidden');
}

document.getElementById('record-btn').addEventListener('click', async () => {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
});

async function startRecording() {
    audioChunks = [];
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            analyzeAudio(audioBlob);
        };
        mediaRecorder.start();
        isRecording = true;
        document.getElementById('record-btn').classList.add('recording');
        document.getElementById('record-status').innerText = "녹음 중... 다시 눌러 완료";
    } catch (err) {
        alert("마이크 접근 권한이 필요합니다.");
    }
}

function stopRecording() {
    mediaRecorder.stop();
    isRecording = false;
    document.getElementById('record-btn').classList.remove('recording');
    document.getElementById('record-status').innerText = "분석 중...";
}

async function analyzeAudio(audioBlob) {
    const targetText = document.getElementById('target-sentence').innerText;
    const formData = new FormData();
    formData.append('audio', audioBlob, 'record.webm');
    formData.append('target_text', targetText);

    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('result-wrapper').classList.add('hidden');

    try:
        // Parallel call for comparison (Step 9) + New Hybrid analysis
        const [acousticRes, whisperRes, hybridRes] = await Promise.all([
            fetch('/api/analyze', { method: 'POST', body: formData }),
            fetch('/api/analyze_whisperx', { method: 'POST', body: formData }),
            fetch('/api/analyze_hybrid', { method: 'POST', body: formData })
        ]);

        const acousticData = await acousticRes.json();
        const whisperData = await whisperRes.json();
        const hybridData = await hybridRes.json();

        displayResults(acousticData, whisperData, hybridData);
    } catch (err) {
        alert("분석 중 오류가 발생했습니다.");
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
}

function displayResults(acoustic, whisper, hybrid) {
    document.getElementById('result-wrapper').classList.remove('hidden');

    // 1. Acoustic Results
    document.getElementById('acoustic-score').innerText = `${acoustic.total_score.toFixed(1)}점`;
    const acousticList = document.getElementById('acoustic-feedback');
    acousticList.innerHTML = '';
    acoustic.feedback_details.forEach(fb => {
        const li = document.createElement('li');
        li.innerText = fb;
        acousticList.appendChild(li);
    });

    // 2. WhisperX Results
    document.getElementById('whisper-score').innerText = `${whisper.total_score.toFixed(1)}점`;
    const whisperList = document.getElementById('whisper-feedback');
    whisperList.innerHTML = '';
    whisper.feedback_details.forEach(fb => {
        const li = document.createElement('li');
        li.innerText = fb;
        whisperList.appendChild(li);
    });

    // 3. Hybrid Summary (Step 5, 6, 7)
    // Update main total score with hybrid results
    const hybridScoreElem = document.getElementById('hybrid-score');
    if (hybridScoreElem) {
        hybridScoreElem.innerText = `${hybrid.total_score.toFixed(1)}점`;
    }
    
    // Add hybrid feedback to a new section or use existing ones
    console.log("Hybrid result:", hybrid);

    // Whisper Alignment Visualization
    const alignmentDiv = document.getElementById('whisper-alignment');
    alignmentDiv.innerHTML = '';
    
    // Check if we have word segments
    if (whisper.analysis_raw && whisper.analysis_raw.length > 0) {
        whisper.analysis_raw.forEach(wordObj => {
            const span = document.createElement('span');
            span.className = 'char-box ' + (wordObj.confidence > 0.7 ? 'high-conf' : 'low-conf');
            span.innerText = wordObj.word;
            span.title = `Confidence: ${(wordObj.confidence * 100).toFixed(1)}%`;
            alignmentDiv.appendChild(span);
        });
    }
}
