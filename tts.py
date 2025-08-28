# tts.py
import pyttsx3
from threading import Lock

_engine = None
_lock = Lock()

def _get_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        # Tune these if you like
        _engine.setProperty('rate', 170)
        _engine.setProperty('volume', 1.0)
    return _engine

def speak(text: str, wait: bool = True):
    """Speak text with pyttsx3 (blocking by default)."""
    with _lock:
        engine = _get_engine()
        engine.say(text)
        if wait:
            engine.runAndWait()
        else:
            # Non-blocking run; usually you want blocking
            engine.startLoop(False)
            engine.iterate()
            engine.endLoop()
