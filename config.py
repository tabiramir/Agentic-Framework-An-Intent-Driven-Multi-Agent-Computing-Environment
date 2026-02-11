# config.py

import os
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

from datetime import timezone, timedelta

from dotenv import load_dotenv

load_dotenv()

# ---- Timezone ----
IST = ZoneInfo("Asia/Kolkata")

# ---- Gmail / Google API ----
GMAIL_SCOPES = [""]

# ---- Audio / VAD / Whisper config ----
SAMPLE_RATE = 16000
MIC_DEVICE_INDEX = None
FRAME_MS = 20
FRAME_LEN = int(SAMPLE_RATE * FRAME_MS / 1000)

VAD_AGGRESSIVENESS = 2
MAX_UTTERANCE_SEC = 20
MIN_UTTERANCE_SEC = 0.8
SILENCE_HANGOVER_MS = 600

WHISPER_MODEL_SIZE = "tiny.en"

# ---- Logs ----
NLU_LOG = Path("nlu_log.jsonl")
AGENT_LOG = Path("agent_log.jsonl")

NLU_LOG.parent.mkdir(parents=True, exist_ok=True)
NLU_LOG.touch(exist_ok=True)
AGENT_LOG.parent.mkdir(parents=True, exist_ok=True)
AGENT_LOG.touch(exist_ok=True)

# --- MongoDB configuration (for logs etc.) ---

MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "agentic_os")

MONGO_COLLECTION_NLU = os.getenv("MONGO_COLLECTION_NLU", "nlu_log")
MONGO_COLLECTION_AGENT = os.getenv("MONGO_COLLECTION_AGENT", "agent_log")
MONGO_COLLECTION_REMINDERS = os.getenv("MONGO_COLLECTION_REMINDERS", "reminders")


# ---- Hotwords ----
HOTWORDS = ["hey agent", "hello agent", "computer", "agentic os"]
SESSION_END_WORDS = ["bye agent", "goodbye agent", "stop listening", "go to sleep"]
HOTWORD_THRESHOLD = 80

# ---- Booking / Google Flights ----
AMADEUS_API_KEY = ""
AMADEUS_API_SECRET = ""
AMADEUS_BASE_URL = ""
# ---- Misc global warnings ----
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")

