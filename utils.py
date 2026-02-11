# utils.py

import atexit
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from config import (
    IST,
    NLU_LOG,
    AGENT_LOG,
    GMAIL_SCOPES,
    MONGO_COLLECTION_NLU,
    MONGO_COLLECTION_AGENT,
)

from db import get_db

_gui_logger = None


# Gmail imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# -------------- TTS / speak() --------------

try:
    import pyttsx3

    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 180)
    _tts_engine.setProperty("volume", 1.0)

    def speak(text: str) -> None:
        """Text-to-speech (with console print)."""
        try:
            print(f"🗣️ {text}")
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception as e:
            print(f"(tts) error: {e}")

except Exception:
    def speak(text: str) -> None:
        """Fallback speak that only prints to console."""
        print(f"🗣️ {text}")


# -------------- Time helpers --------------

def now_local() -> datetime:
    """Return current local time in IST."""
    return datetime.now(IST)


def iso_now() -> str:
    """Return ISO timestamp in IST."""
    return now_local().isoformat()


def make_aware(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware (IST)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


# -------------- Logging (JSONL) --------------

_buffer = []  # buffered NLU logs


def append_jsonl(record: Dict[str, Any]) -> None:
    """Append a record to the NLU JSONL log with small buffering."""
    global _buffer
    if not record:
        return
    _buffer.append(record)
    if len(_buffer) >= 5:
        with NLU_LOG.open("a", encoding="utf-8") as f:
            for r in _buffer:
                # NOTE: default=str handles ObjectId and other non-JSON types
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        _buffer.clear()


def flush_jsonl() -> None:
    """Flush remaining buffered NLU logs."""
    global _buffer
    if not _buffer:
        return
    with NLU_LOG.open("a", encoding="utf-8") as f:
        for r in _buffer:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _buffer = []


atexit.register(flush_jsonl)

def _insert_mongo(collection_name: str, doc: Dict[str, Any]) -> None:
    """Insert a document into MongoDB (best-effort, non-fatal)."""
    if not doc or not collection_name:
        return
    try:
        db = get_db()
        if db is None:
            return
        db[collection_name].insert_one(doc)
    except Exception as e:
        # Don't crash the app if Mongo is down
        print(f"(mongo) insert error into {collection_name}: {e}")

def log_agent(obj: Dict[str, Any]) -> None:
    """Log agent events to AGENT_LOG and MongoDB."""
    if not obj:
        return

    # File log (existing behaviour)
    AGENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AGENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Mongo log (new)
    _insert_mongo(MONGO_COLLECTION_AGENT, obj)



# -------------- Shell / system helpers --------------

def which(programs):
    """Return first found executable from a list (or string)."""
    if isinstance(programs, (list, tuple)):
        for p in programs:
            if shutil.which(p):
                return p
        return None
    return shutil.which(programs)


def open_with(programs, args=None, return_program: bool = False):
    """
    Launch a GUI app with arguments, suppressing stdout/stderr.

    If return_program=False (default):
        → returns bool ok

    If return_program=True:
        → returns (ok: bool, exe: str | None)
    """
    exe = which(programs)
    if not exe:
        if return_program:
            return False, None
        return False

    try:
        if args is None:
            args = []
        subprocess.Popen(
            [exe] + args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"(launcher) launched: {exe} {' '.join(args)}".strip())
        if return_program:
            return True, exe
        return True
    except Exception as e:
        print(f"(launcher) failed {exe}: {e}")
        if return_program:
            return False, None
        return False



def run_cmd(cmd) -> str:
    """Run a command and return its output (or error)."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out
    except subprocess.CalledProcessError as e:
        return e.output
    except Exception as e:
        return str(e)


def expand_dir_keyword(keyword: str) -> Path:
    """
    Map spoken folder keywords like 'downloads' or 'documents'
    to real paths under the user's home.
    """
    home = Path.home()
    mapping = {
        "home": home,
        "downloads": home / "Downloads",
        "documents": home / "Documents",
        "desktop": home / "Desktop",
        "pictures": home / "Pictures",
        "music": home / "Music",
        "videos": home / "Videos",
    }
    if not keyword:
        return home
    kraw = keyword.strip().lower()
    k = kraw.rstrip("s") + ("s" if kraw.endswith("s") else "s")
    for key, path in mapping.items():
        if k.startswith(key.rstrip("s")):
            return path
    return home


def looks_like_url(s: str) -> bool:
    """Rudimentary URL / domain heuristic."""
    s = s.strip().lower()
    return (
        s.startswith("http://")
        or s.startswith("https://")
        or re.match(r"^[a-z0-9\-\.]+\.[a-z]{2,}(/.*)?$", s) is not None
    )


# -------------- Gmail helper --------------

def get_gmail_service():
    """
    Returns an authenticated Gmail API service.
    On first run, opens a browser window for OAuth consent and saves token.json.
    """
    creds = None
    token_path = Path("token.json")

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), GMAIL_SCOPES)

    # Refresh or fetch new token if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"(gmail) token refresh failed: {e}")
                creds = None

        if not creds:
            if not Path("credentials.json").exists():
                print("(gmail) Missing credentials.json for OAuth.")
                return None

            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", GMAIL_SCOPES
            )
            # This opens a local browser for login once
            creds = flow.run_local_server(port=0)

            # Save for next time
            with token_path.open("w", encoding="utf-8") as f:
                f.write(creds.to_json())

    try:
        service = build("gmail", "v1", credentials=creds)
        return service
    except Exception as e:
        print(f"(gmail) Failed to build service: {e}")
        return None
        
        
def log_nlu(record: Dict[str, Any]) -> None:
    """
    Log NLU command both to local JSONL (NLU_LOG)
    and to MongoDB (MONGO_COLLECTION_NLU), if configured.
    """
    if not record:
        return
    append_jsonl(record)
    _insert_mongo(MONGO_COLLECTION_NLU, record)

def attach_gui_logger(logger) -> None:
    """
    Optional hook so GUI (NiceGUI) can receive log lines.
    `logger` is expected to have a .put(str) method (like ThreadLog).
    """
    global _gui_logger
    _gui_logger = logger


def gui_log(msg: str) -> None:
    """
    Log to console AND, if available, to the GUI logger.
    Use this instead of print() when you want messages
    to show up both in terminal and in the NiceGUI console.
    """
    print(msg)
    global _gui_logger
    if _gui_logger is not None:
        try:
            _gui_logger.put(msg)
        except Exception:
            pass

