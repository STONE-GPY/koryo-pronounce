const sentences = ["학교", "수업", "국물 같이 먹자", "고려인", "안녕하세요", "발음 교정", "안녕하세요", "의사 선생님"];
const targetSentenceEl = document.getElementById("target-sentence");
const recordBtn = document.getElementById("record-btn");
const recordStatus = document.getElementById("record-status");
const resultContainer = document.getElementById("result-container");
const loadingContainer = document.getElementById("loading");
const totalScoreEl = document.getElementById("total-score");
const feedbackList = document.getElementById("feedback-list");

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

function changeSentence() {
    const current = targetSentenceEl.innerText;
    let next = current;
    while (next === current) {
        next = sentences[Math.floor(Math.random() * sentences.length)];
    }
    targetSentenceEl.innerText = next;
    resultContainer.classList.add("hidden");
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            // Provide a generic wav/webm mimetype. The backend librosa handles both gracefully if ffmpeg is present.
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            await analyzeAudio(audioBlob);
        };

        mediaRecorder.start();
        isRecording = true;
        recordBtn.classList.add("recording");
        recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
        recordStatus.innerText = "녹음 중... 버튼을 다시 누르면 완료";
        resultContainer.classList.add("hidden");

    } catch (err) {
        alert("마이크 권한이 필요하거나, 브라우저가 지원하지 않습니다.");
        console.error("Mic error:", err);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    isRecording = false;
    recordBtn.classList.remove("recording");
    recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    recordStatus.innerText = "버튼을 눌러 녹음 시작";
}

recordBtn.addEventListener("click", () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

async function analyzeAudio(audioBlob) {
    loadingContainer.classList.remove("hidden");
    resultContainer.classList.add("hidden");

    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    formData.append("target_text", targetSentenceEl.innerText);

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`서버 응답 오류: ${response.status}`);
        }

        const data = await response.json();
        displayResult(data);
    } catch (error) {
        console.error("Analysis error:", error);
        alert("분석 중 오류가 발생했습니다. (백엔드 서버가 켜져 있는지 확인하세요)");
    } finally {
        loadingContainer.classList.add("hidden");
    }
}

function displayResult(data) {
    const score = isNaN(data.total_score) ? 0 : Math.round(data.total_score);
    totalScoreEl.innerText = score;
    
    // Set circle color based on score
    const circle = document.querySelector('.score-circle');
    if (score >= 80) circle.style.borderColor = "var(--success)";
    else if (score >= 50) circle.style.borderColor = "#ffd166";
    else circle.style.borderColor = "var(--danger)";

    feedbackList.innerHTML = "";
    if (data.feedback_details && data.feedback_details.length > 0) {
        data.feedback_details.forEach(fb => {
            const li = document.createElement("li");
            li.innerText = fb;
            feedbackList.appendChild(li);
        });
    } else {
        const li = document.createElement("li");
        li.innerText = "분석된 피드백이 없습니다.";
        feedbackList.appendChild(li);
    }

    resultContainer.classList.remove("hidden");
}