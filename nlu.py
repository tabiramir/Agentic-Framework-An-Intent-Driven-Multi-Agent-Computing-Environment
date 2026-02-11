# nlu.py
"""
NLU module — integrated with gemini_helper for cleaned transcripts and intent refinement.

Behavior:
- For each subcommand we first run the cleaning pipeline
  (gemini_helper.enhance_transcript_sync) and use the cleaned text for intent/entity extraction.
- If gemini_helper.refine_intent_sync is available and GEMINI_ENABLED, we call it to
  get an LLM-refined intent + normalized text and use those values when available,
  but only accept LLM intent when it is reasonably confident.
- If cleaned text yields no intent, fall back to the original raw text for local rules.
"""

import re
from typing import Dict, List, Tuple, Any

import dateparser
import spacy
from rapidfuzz import fuzz, process as fprocess

from config import HOTWORDS, SESSION_END_WORDS
from utils import speak

# Try to import gemini_helper functions (optional)
try:
    import gemini_helper as _gemini
    GEMINI_AVAILABLE = getattr(_gemini, "GEMINI_ENABLED", False) and hasattr(_gemini, "enhance_transcript_sync")
except Exception:
    _gemini = None
    GEMINI_AVAILABLE = False

# --------- Load spaCy ---------
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
except Exception as e:
    print(f"(spacy) model load error: {e}")
    nlp = None
    speak("SpaCy model failed to load; entity extraction may not work.")


# small helper: words -> numbers for common ordinals/cards
_ORDINALS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
}


# --------- Intent refinement / detection ---------

def refine_intent(intent: str, text: str) -> str:
    """
    Legacy rule-based intent refinement (conservative).
    Returns an intent label such as 'app.open', 'file.manage', 'file.manage.missing_filename', etc.
    """
    t = (text or "").lower().strip()

    # --- Booking follow-ups like "open option 1" or "open option one" ---
    # map to booking.search so Planner routes to BookingAgent
    if re.search(r'\b(open|book|go to)\s+(?:the\s+)?(?:option\s+)?\d+\b', t) or \
       re.search(r'\b(open|book|go to)\s+(?:the\s+)?(?:option\s+)?(?:' + '|'.join(_ORDINALS.keys()) + r')\b', t):
        return "booking.search"

    # --- Explicit browser tab controls: prefer these BEFORE generic 'close' ---
    if re.search(r'\b(close|closed|shut)\s+(this\s+|the\s+|current\s+)?tab\b', t):
        return "browser.control"
    if re.search(r'\b(new|open)\s+(a\s+)?tab\b', t) or re.search(r'\bopen\s+new\s+tab\b', t):
        return "browser.control"

    # --- Explicit: open files / file manager -> treat as file manager open ---
    if re.search(r'\b(open|show|browse)\s+(the\s+)?(file manager|files|file explorer|filebrowser|file manager)\b', t):
        return "file.manage"
    if re.search(r'\bopen\s+files\b', t):
        return "file.manage"

    # --- File operations: capture optional filename ---
    m_file = re.search(
        r'\b(?P<verb>open|create|make|new|delete|remove)\s+(?:a\s+)?file(?:\s+(?P<fname>[\w\-\.\' ]+(?:\s+dot\s+[a-z0-9]{1,8})?))?\b',
        t
    )
    if m_file:
        fname = m_file.group("fname")
        if fname:
            return "file.manage"
        else:
            return "file.manage.missing_filename"

    # --- Mail / email intents ---
    if re.search(r'\b(email|emails|mail|mails|inbox|gmail)\b', t) and any(
        w in t for w in ["show me", "check", "latest", "recent", "unread", "open", "read"]
    ):
        return "mail.read"
    if re.search(r'\b(mail|email|message)\b' , t) and any(
        w in t for w in ["loud", "aloud", "speak", "subject"]
    ):
        return "mail.read"

    # Settings
    if (
        re.search(r'\b(open|launch)\s+(system\s*)?settings\b', t)
        or t in {"settings", "system settings"}
    ):
        return "app.open"

    # Booking / commerce intents (search)
    booking_kw = [
        "flight", "flights", "bus", "buses", "train", "trains",
        "movie", "movies", "tickets", "ticket",
        "hotel", "hotels", "stay", "book", "booking", "bookings"
    ]
    if any(k in t for k in booking_kw) or t.startswith("show me the cheapest"):
        return "booking.search"

    # Folder opens → file.manage
    if re.search(
        r'\bopen\s+(?:the\s+)?(home|downloads?|documents?|desktop|pictures?|music|videos?|recent|trash)\s*(folder)?\b',
        t
    ):
        return "file.manage"

    # File ops catch-all
    if any(w in t for w in [
        "open file", "create file", "make file", "new file", "edit file", "delete file", "remove file",
        "open files", "file manager", "files app", "open downloads", "open documents"
    ]):
        return "file.manage"

    if "remind" in t:
        return "reminder.create"

    if re.search(r'\bclose\b.*\bfile\b', t):
        return "file.manage"

    if any(w in t for w in [
        "file manager", "files app", "open files", "open file manager", "create a file",
        "make a file", "new file", "open downloads", "open documents", "open desktop",
        "settings", "open settings", "system settings", "write", "append", "save",
        "delete file", "remove file", "erase file", "trash"
    ]):
        return "file.manage"

    if any(w in t for w in [
        "task manager", "open task manager", "show task manager", "system monitor",
        "which process makes it slow", "top cpu", "top memory", "top ram", "show cpu processes",
        "show memory processes", "slow processes", "high cpu", "high memory", "why is it slow", "lag"
    ]):
        return "process.monitor"

    if any(w in t for w in [
        "don't sleep", "dont sleep", "keep awake", "keep running", "prevent sleep",
        "caffeinate", "stay awake", "keep system awake", "no sleep",
        "allow sleep", "stop preventing sleep", "let it sleep",
        "disable keep awake", "stop keep awake"
    ]):
        return "sleep.control"

    if any(w in t for w in [
        "new tab", "close tab", "next tab", "previous tab", "prev tab", "back", "forward",
        "scroll down", "scroll up", "scroll to top", "scroll to bottom",
        "go to ", "open url", "open website", "focus address bar", "address bar",
        "browser search", "type in address bar"
    ]):
        return "browser.control"

    # Generic web search
    if any(w in t for w in ["search", "find", "look up", "google", "web search"]) and "browser search" not in t:
        return "web.search"

    # default app open
    if "open" in t or "launch" in t:
        return "app.open"

    if "play" in t:
        return "music.play"

    # Now generic single-word/short 'close' -> fallback to 'close' (app close)
    if any(w in t for w in [
        "close", "quit", "exit", "force close", "kill process", "stop"
    ]):
        return "close"

    return intent


def detect_intent(text: str) -> Tuple[str, float]:
    """
    Detect intent with the following preference:
      1) local rule (refine_intent) on given text
      2) if that is 'unknown' and Gemini available, try LLM refine
      3) otherwise, return local label (may be 'unknown')
    """
    t_in = (text or "").strip()
    if not t_in:
        return "unknown", 0.0

    # 1) local rules first
    try:
        local_label = refine_intent("unknown", t_in)
    except Exception:
        local_label = "unknown"

    if local_label and local_label != "unknown":
        return local_label, 1.0

    # 2) local could not decide => try gemini refine (if available)
    try:
        from gemini_helper import refine_intent_sync, GEMINI_ENABLED as _G_ENABLED
        if _G_ENABLED and callable(refine_intent_sync):
            try:
                res = refine_intent_sync(t_in)
                if isinstance(res, dict):
                    intent = res.get("intent") or "unknown"
                    conf = float(res.get("confidence", 0.0))
                    if intent and intent != "unknown" and conf > 0.1:
                        return intent, max(0.0, min(1.0, conf))
            except Exception:
                pass
    except Exception:
        pass

    # fallback
    return local_label, 0.0 if local_label == "unknown" else 1.0


# --------- Entities / normalization ---------

def extract_entities(text: str) -> List[Dict[str, str]]:
    if not nlp:
        return []
    try:
        return [{"label": e.label_, "text": e.text} for e in nlp(text).ents]
    except Exception:
        return []


def split_into_subcommands(sentence: str) -> List[str]:
    s = (sentence or "").strip()
    parts = re.split(r'\s+(?:and|then|,)\s+', s, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def _parse_option_index(text: str) -> int | None:
    """
    Return integer option index if found (1-based), else None.
    Handles both digits and common ordinal/word forms.
    """
    if not text:
        return None
    # digits first
    m = re.search(r'\boption\s+(\d+)\b', text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    # words / ordinals
    for w, v in _ORDINALS.items():
        if re.search(r'\b(?:option\s+)?' + re.escape(w) + r'\b', text, re.IGNORECASE):
            return v
    return None


def normalize_entities(entities: List[Dict[str, str]], text: str) -> Dict[str, str]:
    norm: Dict[str, str] = {}
    t_raw = (text or "").strip()
    lower = t_raw.lower()

    # Date / time
    times = [
        dateparser.parse(e["text"])
        for e in entities
        if e.get("label") in ("DATE", "TIME") and dateparser.parse(e["text"])
    ]
    if times:
        norm["datetime"] = times[0].isoformat()

    # App names
    for kw in [
        "chrome", "calculator", "terminal", "spotify", "vscode", "firefox",
        "settings", "music", "vlc", "rhythmbox", "code",
        "gnome-calculator", "galculator", "kcalc"
    ]:
        if kw in lower:
            norm["application"] = (
                "calculator"
                if kw in ("gnome-calculator", "galculator", "kcalc", "calculator")
                else kw
            )
            break

    # File names: assemble spoken "name dot ext" into name.ext
    m_dot = re.search(r"([\w\-\s']+?)\s+(?:dot|period|\.)\s+([a-z0-9]{1,8})\b", t_raw, re.IGNORECASE)
    if m_dot:
        base = m_dot.group(1).strip().replace(" ", "_")
        ext = m_dot.group(2).strip()
        filename = f"{base}.{ext}"
        norm["file"] = filename
    else:
        for token in re.findall(r"[\w\-\.\']+", t_raw):
            if any(token.lower().endswith(ext) for ext in [
                ".txt", ".pdf", ".docx", ".csv", ".md", ".py", ".json", ".yaml", ".yml"
            ]):
                norm["file"] = token.strip("'\"")
                break

    # Directories
    for dir_kw in ["downloads", "documents", "desktop", "pictures", "music", "videos", "home"]:
        if dir_kw in lower:
            norm["directory"] = dir_kw
            break

    # Web search query
    if any(w in lower for w in ["search", "find", "look up", "google", "web search"]):
        q = t_raw
        for w in ["search for", "search", "find", "look up", "google", "web search", "on the web", "in browser"]:
            q = q.replace(w, "")
        norm["search_query"] = q.strip()

    # Go-to target (URL or domain)
    m = re.search(r"(?:go to|open url|open website)\s+(.+)$", t_raw, re.IGNORECASE)
    if m:
        raw_target = m.group(1).strip()

        # Basic cleanup: collapse repeated spaces, normalize ' dot ' -> '.'
        s = re.sub(r'\s+', ' ', raw_target).strip()
        # replace spoken "dot" with '.' (handle 'dot' and when user says 'dot com')
        s = re.sub(r'\b(dot|period)\b', '.', s, flags=re.IGNORECASE)

        # Remove spaces around dots
        s = re.sub(r'\s*\.\s*', '.', s)

        # If it's multiple words with no dots, try to collapse likely hostname words:
        parts = s.split()
        if len(parts) == 1:
            candidate = parts[0]
        else:
            if all(re.fullmatch(r'[A-Za-z0-9\-]+', p) for p in parts) and len(parts) <= 3:
                candidate = ''.join(parts)
            else:
                candidate = s  # multi-word, likely search phrase; leave as-is

        cand = candidate.strip()
        if re.search(r'\.', cand):
            cand = cand.strip('. ')
            norm["goto_target"] = cand
        else:
            if re.search(r'\b(com|org|net|io|co|in)\b', raw_target, re.IGNORECASE):
                cand2 = re.sub(r'\b(com|org|net|io|co|in)\b', '', cand, flags=re.IGNORECASE).strip()
                if cand2:
                    cand = cand2 + '.com'
                norm["goto_target"] = cand
            else:
                if re.fullmatch(r'[A-Za-z0-9\-]{2,30}', cand):
                    cand_with_com = cand + '.com'
                    norm["goto_target"] = cand_with_com
                else:
                    norm["goto_target"] = s

    # Browser actions (explicit)
    if re.search(r'\b(close|closed|shut)\s+(this\s+|the\s+|current\s+)?tab\b', lower):
        norm["browser_action"] = "close_tab"
    elif re.search(r'\b(new|open)\s+(a\s+)?tab\b', lower) or re.search(r'\bopen\s+new\s+tab\b', lower):
        norm["browser_action"] = "new_tab"

    # Booking option extraction: "open option 1" or "open first option"
    opt = _parse_option_index(t_raw)
    if opt is not None:
        # store as 1-based index
        norm["option"] = int(opt)

    # Quick "write/append" pattern
    m_qwrite = re.search(
        r'(?:write|append|add|create|make)\s+(?P<content>"[^"]+"|\'[^\']+\')\s+(?:to|in|into)\s+(?P<target>[^,;]+)',
        lower,
    )
    if m_qwrite:
        norm["content"] = m_qwrite.group("content")[1:-1]
        tgt = m_qwrite.group("target").strip().strip('\'"')
        if tgt in {"home", "downloads", "documents", "desktop", "pictures", "music", "videos"}:
            norm["directory"] = tgt
        else:
            norm["file"] = tgt

    return norm


# --------- Hotword / endword detection ---------

def hotword_detect(text: str):
    if not text.strip():
        return None, 0
    cand, score, *_ = fprocess.extractOne(
        text.lower(),
        HOTWORDS,
        scorer=fuzz.token_sort_ratio,
    )
    return cand, score


def endword_detect(text: str):
    if not text.strip():
        return None, 0
    cand, score, *_ = fprocess.extractOne(
        text.lower(),
        SESSION_END_WORDS,
        scorer=fuzz.token_sort_ratio,
    )
    return cand, score


# --------- High-level command helpers ---------

def build_command(text: str) -> Dict[str, Any]:
    """
    Build a normalized command dict from raw text.
    Uses gemini_helper.enhance_transcript_sync (if available) to produce cleaned_text,
    then runs detect_intent/extract_entities/normalize_entities on the cleaned_text.
    If cleaned_text yields unknown intent, fall back to running detect_intent on the original text.
    """
    text = (text or "").strip()
    if not text:
        return {}

    # Prefer gemini to clean the transcript (aggressive mode)
    cleaned = text
    try:
        if GEMINI_AVAILABLE and hasattr(_gemini, "enhance_transcript_sync"):
            cleaned = _gemini.enhance_transcript_sync(text)
        else:
            cleaned = text
    except Exception:
        cleaned = text

    # 1) Try intent on cleaned text
    intent_label, conf = detect_intent(cleaned)

    # 2) If cleaned gives unknown, fallback to original (conservative)
    if intent_label == "unknown":
        try:
            orig_label, orig_conf = detect_intent(text)
            # if original has a real label, prefer it
            if orig_label and orig_label != "unknown":
                intent_label, conf = orig_label, orig_conf
        except Exception:
            pass

    # 3) Allow gemini intent refinement override if provided AND confident
    normalized_from_gemini = None
    try:
        if GEMINI_AVAILABLE and hasattr(_gemini, "refine_intent_sync"):
            try:
                gi = _gemini.refine_intent_sync(cleaned)
                if isinstance(gi, dict):
                    gi_intent = gi.get("intent")
                    gi_conf = float(gi.get("confidence", 0.0)) if gi.get("confidence") is not None else None
                    gi_norm_text = gi.get("normalized_text")
                    if gi_intent and gi_intent != "unknown" and (gi_conf and gi_conf > 0.1):
                        # only accept gemini's intent if it claims >= 0.1 confidence
                        intent_label = gi_intent
                        conf = max(0.0, min(1.0, float(gi_conf or conf)))
                    if gi_norm_text:
                        normalized_from_gemini = gi_norm_text
            except Exception:
                pass
    except Exception:
        pass

    # Entities extracted from gemini-normalized text (if present) OR cleaned/original
    entity_source_text = normalized_from_gemini or cleaned or text
    entities = extract_entities(entity_source_text)
    norm = normalize_entities(entities, entity_source_text)

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "intent": {"label": intent_label, "confidence": round(conf, 3)},
        "entities": entities,
        "normalized": norm,
        "tokens": (entity_source_text or "").split(),
    }


def process_text_commands(utterance: str) -> List[Dict[str, Any]]:
    """
    Top-level NLU entry:
      - splits an utterance into subcommands (using 'and', 'then', commas)
      - runs build_command() on each cleaned piece (cleaning happens inside build_command).
    """
    base = (utterance or "").strip()
    if not base:
        return []

    parts = split_into_subcommands(base)
    if not parts:
        parts = [base]

    cmds: List[Dict[str, Any]] = []
    for p in parts:
        cmd = build_command(p)
        if cmd:
            cmds.append(cmd)

    return cmds

