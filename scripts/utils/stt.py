# stt.py
import os
import queue
import json
import time
import sounddevice as sd
import vosk

MODEL_PATH = r"C:\Users\Admin\Downloads\interview_assistant\models\vosk-model-small-en-us-0.15"

q = queue.Queue()

def _callback(indata, frames, time_info, status):
    if status:
        print("Status:", status)
    q.put(bytes(indata))

def init_vosk(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Vosk model not found at: {model_path}")
    return vosk.Model(model_path)

vosk_model = init_vosk()

def transcribe(timeout: int = 10, samplerate: int = 16000):
    """Listen for up to `timeout` seconds and return the transcription."""
    print(f"Listening for up to {timeout} seconds...")
    results = []
    try:
        with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                               channels=1, callback=_callback):
            rec = vosk.KaldiRecognizer(vosk_model, samplerate)
            start = time.time()
            while time.time() - start < timeout:
                data = q.get()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if res.get("text"):
                        results.append(res["text"])
            # Final flush
            final = json.loads(rec.FinalResult())
            if final.get("text"):
                results.append(final["text"])
    except Exception as e:
        print("STT error:", e)
        return ""
    text = " ".join(results).strip()
    print("You said:", text)
    return text
