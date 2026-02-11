# gemini_helper.py
"""
Gemini post-processing helper for ASR transcripts.

Provides:
- apply_simple_corrections (token-aware)
- _local_fuzzy_fix (token-level fuzzy replacements)
- enhance_transcript_sync / enhance_transcript_async
- rerank_candidates_sync
- refine_intent_sync

Safe: works without google.generativeai installed (falls back to local rules).
"""

import os
import re
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, List, Dict, Any

# Optional import of google generative ai client
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

# -------------------------
# Config via env (tunable)
# -------------------------
GEMINI_ENABLED = os.getenv("GEMINI_ENABLED", "true").lower() in ("1", "true", "yes")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_MAX_CONCURRENT = int(os.getenv("GEMINI_MAX_CONCURRENT", "3"))
GEMINI_CACHE_TTL = int(os.getenv("GEMINI_CACHE_TTL", "300"))
GEMINI_CALL_TIMEOUT = float(os.getenv("GEMINI_CALL_TIMEOUT", "5.0"))

if GEMINI_ENABLED and genai and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

# -------------------------
# Local corrections
# -------------------------
try:
    from rapidfuzz import process as _fproc, fuzz as _fuzz
except Exception:
    _fproc = None
    _fuzz = None

_PROTECTED_TOKENS = {
    "hey", "agent", "open", "create", "make", "new", "file", "files",
    "notes", "note", "dot", "txt", "pdf", "docx",
    "browser", "calculator", "gmail", "email", "search",
}

_SIMPLE_CORRECTIONS = {
    "bowser": "browser",
    "oppen": "open",
    "serch": "search",
    "chepest": "cheapest",
    "flites": "flights",
    "flytes": "flights",
    "tomaro": "tomorrow",
    "tommorow": "tomorrow",
}

_SIMPLE_PATTERNS = [
    (re.compile(r'\b' + re.escape(k) + r'\b', re.I), v)
    for k, v in _SIMPLE_CORRECTIONS.items()
]

# -------------------------
# 🔐 Gemini keyword protection (FIX)
# -------------------------
_GEMINI_PROTECTED_WORDS = [
    "hotel", "hotels",
    "flight", "flights",
    "bus", "buses",
    "train", "trains",
    "file", "files",
    "folder", "folders",
    "website", "websites",
    "url", "link",
    "browser", "calculator",
]

def _freeze_keywords_for_gemini(text: str):
    placeholders = {}
    frozen = text

    for idx, word in enumerate(_GEMINI_PROTECTED_WORDS):
        pattern = r"\b" + re.escape(word) + r"\b"

        def repl(m):
            key = f"__KW_{idx}__"
            placeholders[key] = m.group(0)
            return key

        frozen = re.sub(pattern, repl, frozen, flags=re.IGNORECASE)

    return frozen, placeholders


def _restore_keywords_from_gemini(text: str, placeholders: Dict[str, str]) -> str:
    restored = text
    for key, original in placeholders.items():
        restored = restored.replace(key, original)
    return restored


# -------------------------
# Correction helpers
# -------------------------
def apply_pre_corrections(text: str) -> str:
    if not text:
        return text
    return re.sub(r'\s+', ' ', text).strip()


def apply_simple_corrections(text: str) -> str:
    if not text:
        return text

    tokens = re.findall(r"\w+|[^\w\s]", text)
    out = []

    for tok in tokens:
        low = tok.lower()

        if low in _PROTECTED_TOKENS or not low.isalpha():
            out.append(tok)
            continue

        if low in _SIMPLE_CORRECTIONS:
            repl = _SIMPLE_CORRECTIONS[low]
            out.append(repl.capitalize() if tok[0].isupper() else repl)
            continue

        replaced = False
        for pat, repl in _SIMPLE_PATTERNS:
            if pat.fullmatch(tok):
                out.append(repl)
                replaced = True
                break

        if not replaced:
            out.append(tok)

    s = " ".join(out)
    return re.sub(r'\s+([.,!?])', r'\1', s)


def _local_fuzzy_fix(text: str, threshold: int = 70) -> str:
    if not (_fproc and _fuzz):
        return text

    tokens = re.findall(r"\w+|[^\w\s]", text)
    out = []

    candidates = list(_SIMPLE_CORRECTIONS.values())

    for tok in tokens:
        if not tok.isalpha() or tok.lower() in _PROTECTED_TOKENS:
            out.append(tok)
            continue

        match = _fproc.extractOne(tok, candidates, scorer=_fuzz.ratio)
        if match and match[1] >= threshold:
            out.append(match[0])
        else:
            out.append(tok)

    return " ".join(out)


# -------------------------
# Prompts
# -------------------------
_SYSTEM_PROMPT_CLEAN = (
    "You are a speech post-processor. Only fix spelling and spacing. "
    "Do not change intent. Do not replace domain words. "
    "Return a single cleaned line."
)

# -------------------------
# Caching + concurrency
# -------------------------
class _TTLCache:
    def __init__(self, ttl=300):
        self.ttl = ttl
        self._data = {}
        self._lock = threading.Lock()

    def get(self, k):
        with self._lock:
            v = self._data.get(k)
            if not v:
                return None
            val, ts = v
            if time.time() - ts > self.ttl:
                del self._data[k]
                return None
            return val

    def set(self, k, v):
        with self._lock:
            self._data[k] = (v, time.time())


_CACHE = _TTLCache()
_EXEC = ThreadPoolExecutor(max_workers=GEMINI_MAX_CONCURRENT)
_SEM = threading.Semaphore(GEMINI_MAX_CONCURRENT)


def _safe_call_gemini_sync(prompt: str, timeout: float):
    if not (genai and GEMINI_API_KEY):
        raise RuntimeError("Gemini unavailable")

    with _SEM:
        model = genai.get_model(GEMINI_MODEL)
        resp = model.generate(input=prompt, temperature=0.0, max_output_tokens=128)
        return (resp.text or "").strip()


# -------------------------
# MAIN ENTRY (FIXED)
# -------------------------
def enhance_transcript_sync(raw_text: str, use_cache: bool = True) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return raw_text

    cached = _CACHE.get(raw_text)
    if cached:
        return cached

    pre = apply_pre_corrections(raw_text)
    local = apply_simple_corrections(pre)
    local_fuzzy = _local_fuzzy_fix(local)

    final = local_fuzzy
    used_gemini = False

    if GEMINI_ENABLED and genai and GEMINI_API_KEY:
        try:
            frozen, placeholders = _freeze_keywords_for_gemini(raw_text)
            prompt = _SYSTEM_PROMPT_CLEAN + "\n\nUser transcript:\n" + frozen
            cleaned = _safe_call_gemini_sync(prompt, GEMINI_CALL_TIMEOUT)

            if cleaned:
                final = _restore_keywords_from_gemini(cleaned, placeholders)
                used_gemini = True
        except Exception:
            final = local_fuzzy

    _CACHE.set(raw_text, final)
    return final


def enhance_transcript_async(raw_text: str, use_cache: bool = True) -> Future:
    return _EXEC.submit(enhance_transcript_sync, raw_text, use_cache)

