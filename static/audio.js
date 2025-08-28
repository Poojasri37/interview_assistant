let mediaRecorder;
let chunks = [];
let interviewing = false;

async function startInterview() {
  interviewing = true;
  document.getElementById("status").innerText = "Interview started";
  await askNextQuestion();
}

async function askNextQuestion() {
  const res = await fetch("/candidate/next_question");
  const data = await res.json();

  if (data.done) {
    interviewing = false;
    document.getElementById("question").innerText = "";
    document.getElementById("system").innerText = data.message;
    window.location.href = "/candidate/finish";
    return;
  }

  const q = data.question;
  document.getElementById("question").innerText = q;
  document.getElementById("system").innerText = "Speak now (10 seconds)...";

  // TTS (avatar speaks)
  const speech = new SpeechSynthesisUtterance(q);
  speech.lang = "en-US";

  speech.onend = async () => {
    await record10Seconds();
  };

  window.speechSynthesis.speak(speech);
}

async function record10Seconds() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  chunks = [];

  mediaRecorder.ondataavailable = e => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  mediaRecorder.onstop = async () => {
    const blob = new Blob(chunks, { type: "audio/webm" });
    await sendAudio(blob);
    stream.getTracks().forEach(t => t.stop());
  };

  mediaRecorder.start();
  setTimeout(() => {
    mediaRecorder.stop();
  }, 10000);
}

async function sendAudio(blob) {
  document.getElementById("status").innerText = "Transcribing...";
  const form = new FormData();
  form.append("audio", blob, "answer.webm");

  const res = await fetch("/candidate/answer", {
    method: "POST",
    body: form
  });

  const data = await res.json();

  if (data.done) {
    document.getElementById("system").innerText = data.response || "Finished.";
    window.location.href = "/candidate/finish";
    return;
  }

  if (data.action === "retry") {
    document.getElementById("system").innerText = data.response || "Please repeat.";
    await askNextQuestion();
    return;
  }

  if (data.action === "explain") {
    const speech = new SpeechSynthesisUtterance(data.response);
    speech.lang = "en-US";
    speech.onend = async () => {
      await askNextQuestion();
    };
    window.speechSynthesis.speak(speech);
  } else {
    document.getElementById("system").innerText = data.response || "OK, next.";
    await askNextQuestion();
  }
}
