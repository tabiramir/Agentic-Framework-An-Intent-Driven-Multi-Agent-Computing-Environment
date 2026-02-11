# audio.py

import queue
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import webrtcvad
import whisper

from config import (
    SAMPLE_RATE,
    FRAME_MS,
    FRAME_LEN,
    VAD_AGGRESSIVENESS,
    SILENCE_HANGOVER_MS,
    MAX_UTTERANCE_SEC,
    MIN_UTTERANCE_SEC,
    WHISPER_MODEL_SIZE,
)
from utils import speak

# --------- Microphone selection ---------


def get_default_mic() -> Optional[int]:
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                print(f"🎤 Using mic device: {i} → {dev['name']}")
                return i
    except Exception as e:
        print(f"⚠️ Could not query audio devices: {e}")
    print("⚠️ No microphone found, using default")
    return None


MIC_DEVICE_INDEX = get_default_mic()

# --------- Audio streaming ---------


class AudioStream:
    def __init__(self, samplerate=SAMPLE_RATE, device=MIC_DEVICE_INDEX, channels=1):
        self.samplerate = samplerate
        self.device = device
        self.channels = channels
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, t, status):
        try:
            if indata is None:
                return
            if getattr(indata, "ndim", 1) > 1 and indata.shape[1] > 1:
                mono = np.mean(indata, axis=1)
            else:
                mono = indata[:, 0] if getattr(indata, "ndim", 1) > 1 else indata
            self.q.put(np.asarray(mono, dtype=np.float32).copy())
        except Exception:
            pass

    def start(self):
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            device=self.device,
            channels=self.channels,
            dtype="float32",
            blocksize=FRAME_LEN,
            callback=self._callback,
        )
        self.stream.start()

    def read_frame(self) -> Tuple[np.ndarray, bytes]:
        data = self.q.get(timeout=1)
        pcm16 = (np.clip(data, -1, 1) * 32767).astype(np.int16).tobytes()
        return data, pcm16

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None


# --------- VAD segmentation ---------


class VADSegmenter:
    def __init__(
        self,
        samplerate=SAMPLE_RATE,
        vad_level=VAD_AGGRESSIVENESS,
        frame_len=FRAME_LEN,
        hangover_ms=SILENCE_HANGOVER_MS,
    ):
        self.vad = webrtcvad.Vad(vad_level)
        self.samplerate = samplerate
        self.frame_len = frame_len
        self.hangover_frames = int(hangover_ms / FRAME_MS)
        self.reset()

    def reset(self):
        self.buffer_pcm = bytearray()
        self.buffer_f32 = []
        self.active = False
        self.silence_count = 0

    def push(self, pcm_bytes: bytes, f32_frame: np.ndarray):
        try:
            is_speech = self.vad.is_speech(pcm_bytes, self.samplerate)
        except Exception:
            is_speech = False

        if is_speech:
            self.buffer_pcm.extend(pcm_bytes)
            self.buffer_f32.append(f32_frame)
            self.silence_count = 0
            self.active = True
        else:
            if self.active:
                self.silence_count += 1
                self.buffer_pcm.extend(pcm_bytes)
                self.buffer_f32.append(f32_frame)

        out = None
        if self.active:
            dur = sum(len(f) for f in self.buffer_f32) / self.samplerate
            if (self.silence_count >= self.hangover_frames and dur >= MIN_UTTERANCE_SEC) or dur >= MAX_UTTERANCE_SEC:
                out = np.concatenate(self.buffer_f32) if self.buffer_f32 else np.array([], dtype=np.float32)
                self.reset()
        return out


# --------- Whisper transcription ---------

print("Loading Whisper + NLP models…")
try:
    _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
except Exception as e:
    print(f"(whisper) model load error: {e}")
    _whisper_model = None
    speak("Whisper model failed to load; speech recognition will not work.")


def transcribe_numpy(audio_f32: np.ndarray):
    if audio_f32 is None or len(audio_f32) == 0:
        return "", "en", {}
    if _whisper_model is None:
        return "", "en", {}
    try:
        result = _whisper_model.transcribe(
            audio_f32,
            fp16=False,
            task="transcribe",
            language="en",
        )
        return result.get("text", "").strip(), result.get("language", "en"), result
    except Exception as e:
        print(f"(whisper) error: {e}")
        return "", "en", {}

