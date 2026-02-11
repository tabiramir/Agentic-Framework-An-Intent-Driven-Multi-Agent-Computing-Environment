# main.py

# -----------------------------------------
# Load .env BEFORE all other imports
# -----------------------------------------
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# -----------------------------------------
# Standard imports AFTER .env is loaded
# -----------------------------------------
import queue
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import config
from audio import AudioStream, VADSegmenter, transcribe_numpy
from nlu import hotword_detect, endword_detect, process_text_commands
from utils import iso_now, speak, log_nlu
from agents.planner import Planner

# -----------------------------------------
# Gemini import AFTER env vars loaded
# -----------------------------------------
try:
    from gemini_helper import enhance_transcript_async
    _GEMINI_AVAILABLE = True
except Exception as e:
    print(f"(init) Gemini helper failed to import: {e}")
    enhance_transcript_async = None
    _GEMINI_AVAILABLE = False

print(f"(init) GEMINI_AVAILABLE={_GEMINI_AVAILABLE}, GEMINI_ENABLED={os.getenv('GEMINI_ENABLED')}")

# Default settings
SAMPLE_RATE = getattr(config, "SAMPLE_RATE", 16000)
MIC_DEVICE_INDEX = getattr(config, "MIC_DEVICE_INDEX", None)
HOTWORD_THRESHOLD = getattr(config, "HOTWORD_THRESHOLD", 80)
GEMINI_WAIT_TIMEOUT = float(os.getenv("GEMINI_WAIT_TIMEOUT", "2.0"))


# -----------------------------------------
# Main Audio Loop
# -----------------------------------------
def main():
    audio = AudioStream()
    vad = VADSegmenter()
    planner = Planner()

    try:
        audio.start()
    except Exception as e:
        print(f"[ERROR] Failed to start audio input: {e}")
        speak("Audio input failed to start.")
        return

    mode = "idle"
    print("🎧 Say 'hey agent' to start a session. Say 'bye agent' to stop listening.")
    speak("Agent initialized and sleeping. Say hey agent to wake me.")

    try:
        while True:
            # ---------------------------------
            # READ AUDIO FRAME
            # ---------------------------------
            try:
                f32, pcm = audio.read_frame()
            except queue.Empty:
                continue
            except Exception:
                continue

            # ---------------------------------
            # VAD segmentation
            # ---------------------------------
            seg = vad.push(pcm, f32)
            if seg is None:
                continue

            # ---------------------------------
            # Whisper transcription
            # ---------------------------------
            utt = seg.astype(np.float32)
            text, _, _ = transcribe_numpy(utt)
            if not text:
                continue

            # -----------------------------------------
            # GEMINI POST-PROCESSING (non-blocking)
            # -----------------------------------------
            if _GEMINI_AVAILABLE and enhance_transcript_async is not None:
                try:
                    print("(gemini) scheduling enhancement...")
                    fut = enhance_transcript_async(text)

                    try:
                        cleaned = fut.result(timeout=GEMINI_WAIT_TIMEOUT)
                    except Exception as e:
                        print(f"(gemini) timeout/error: {e}")
                        cleaned = None

                    if cleaned and cleaned.strip() and cleaned.strip() != text.strip():
                        print(f"[Whisper] {text}")
                        print(f"[Gemini]  {cleaned}")
                        text = cleaned
                    else:
                        print("(gemini) no change.")
                        print(f"[Heard] {text}")

                except Exception as e:
                    print(f"(gemini) post-process failed: {e}")
                    print(f"[Heard] {text}")

            else:
                print("(gemini) disabled or unavailable.")
                print(f"[Heard] {text}")

            # -----------------------------------------
            # HOTWORD HANDLING
            # -----------------------------------------
            if mode == "idle":
                cand, score = hotword_detect(text)
                if cand and score >= HOTWORD_THRESHOLD:
                    print(f"🟢 Session started with: {cand}")
                    speak("I'm listening.")
                    try:
                        planner.sleep.keep_awake()
                    except Exception:
                        pass
                    mode = "session"
                continue

            # -----------------------------------------
            # SESSION MODE
            # -----------------------------------------
            if mode == "session":
                end_cand, end_score = endword_detect(text)
                if end_cand and end_score >= HOTWORD_THRESHOLD:
                    print(f"🔴 Session ended with: {end_cand}")
                    speak("Going to sleep.")
                    try:
                        planner.sleep.allow_sleep()
                    except Exception:
                        pass
                    mode = "idle"
                    continue

                # NLU + Planner
                cmds = process_text_commands(text)
                if not cmds:
                    continue

                for partial in cmds:
                    full_cmd = {
                        "module": "speech_nlu",
                        "ts": iso_now(),
                        **partial,
                    }

                    try:
                        log_nlu(full_cmd)
                    except Exception as e:
                        print(f"(log) log_nlu error: {e}")

                    try:
                        planner.handle(full_cmd)
                    except Exception as e:
                        print(f"[ERROR] Planner error: {e}")
                        speak("I ran into an error while handling that.")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try:
            audio.stop()
        except Exception:
            pass
        print("Audio stopped. Bye.")
        speak("Shutting down audio. Bye.")


# -----------------------------------------
# ENTRY POINT
# -----------------------------------------
if __name__ == "__main__":
    main()

