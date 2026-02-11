# app_nicegui.py

import contextlib
import io
import json
import queue
import random
import sys
import threading
import time
import traceback
# Load .env early so gemini_helper and other modules see env vars
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # fallback to automatic discovery
    load_dotenv()

from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import sounddevice as sd
from nicegui import ui, app

from nlu import process_text_commands
from utils import log_nlu, iso_now, get_gmail_service, attach_gui_logger

# =========================
# Import core (from main.py)
# =========================
try:
    import main as core
except ModuleNotFoundError:
    print("Couldn't import 'main.py'. Make sure it's in the same folder.")
    raise
    
# Try importing gemini helpers for typed-input enhancement (optional)
try:
    from gemini_helper import enhance_transcript_sync, enhance_transcript_async
    _GEMINI_FOR_TYPED = True
except Exception as _e:
    enhance_transcript_sync = None
    enhance_transcript_async = None
    _GEMINI_FOR_TYPED = False

print(f"(init) GEMINI_FOR_TYPED={_GEMINI_FOR_TYPED}, GEMINI_ENABLED={os.getenv('GEMINI_ENABLED')}")


# =========================
# Thread-safe log collector
# =========================
class ThreadLog:
    def __init__(self):
        self._q = queue.Queue()

    def put(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._q.put(f"[{ts}] {msg}")

    def drain(self, max_items: int = 500) -> List[str]:
        out = []
        try:
            while len(out) < max_items:
                out.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return out


# =========================
# Stream redirect (mirror)
# =========================
class _StreamToThreadLog(io.TextIOBase):
    """
    Send output to:
      - the original terminal stream (mirror)
      - the shared ThreadLog (for the web UI)
    So:
      - terminal behaves as usual
      - web console shows the same lines with minimal delay
    """

    def __init__(self, logger: ThreadLog, prefix: str = "", mirror=None):
        super().__init__()
        self.logger = logger
        self.prefix = prefix
        self._buf = ""
        self.mirror = mirror  # e.g. sys.__stdout__ / sys.__stderr__

    def write(self, s: str):
        if not isinstance(s, str):
            try:
                s = s.decode(errors="ignore")
            except Exception:
                s = str(s)

        # Mirror to terminal immediately
        if self.mirror is not None:
            try:
                self.mirror.write(s)
                self.mirror.flush()
            except Exception:
                pass

        # Buffer lines into logger
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                self.logger.put(f"{self.prefix}{line}")
        return len(s)

    def flush(self):
        if self._buf.strip():
            self.logger.put(f"{self.prefix}{self._buf.strip()}")
        self._buf = ""


# =========================
# Background listener
# =========================
@dataclass
class BgState:
    mode: str = "idle"  # "idle" | "session"
    last_text: str = ""
    last_nlu: dict | None = None
    listening: bool = False
    hotword_threshold: int = getattr(core, "HOTWORD_THRESHOLD", 80)
    level: float = 0.0  # 0..1: audio level for Siri animation
    vad_active: bool = False  # whether VAD is currently active


class BackgroundListener:
    """
    Owns audio/VAD/Whisper loop.
    Runs in a thread; no direct UI calls.
    All prints are mirrored to terminal + ThreadLog.
    """

    def __init__(self, samplerate: int, mic_index: Optional[int], logger: ThreadLog):
        self.samplerate = samplerate
        self.mic_index = mic_index
        self.logger = logger

        self.state = BgState()
        self.thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        self.planner = core.Planner()
        self.audio_stream = None
        self.vad = None

    def _log(self, s: str):
        self.logger.put(s)

    def is_running(self) -> bool:
        return self.state.listening and self.thread and self.thread.is_alive()

    def start(self):
        if self.is_running():
            return
        self._stop_flag.clear()
        self.thread = threading.Thread(
            target=self._loop,
            name="agentic_os_nicegui_loop",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self._stop_flag.set()
        try:
            if self.audio_stream:
                self.audio_stream.stop()
        except Exception:
            pass
        if self.thread:
            self.thread.join(timeout=2.0)
        self.state.listening = False

    def _loop(self):
        # Capture all core prints → terminal + web log
        out = _StreamToThreadLog(self.logger, mirror=sys.__stdout__)
        err = _StreamToThreadLog(self.logger, prefix="[stderr] ", mirror=sys.__stderr__)

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                # Audio config in this thread
                try:
                    sd.default.samplerate = self.samplerate
                    sd.default.device = (self.mic_index, None)
                except Exception as e:
                    self._log(f"(audio) Failed to set device {self.mic_index}: {e}")

                self.audio_stream = core.AudioStream(
                    samplerate=self.samplerate,
                    device=self.mic_index,
                    channels=1,
                )
                self.vad = core.VADSegmenter(samplerate=self.samplerate)

                try:
                    self.audio_stream.start()
                    self._log("🎙️ Audio stream started.")
                except Exception as e:
                    self._log(f"[ERROR] Failed to start audio input: {e}")
                    return

                self.state.listening = True

                # IMPORTANT CHANGE:
                # If text already woke the agent (mode == "session"),
                # do NOT force it back to idle on audio start.
                if self.state.mode != "session":
                    self.state.mode = "idle"
                    self._log("Agent idle. Say 'hey agent' to wake it.")
                else:
                    self._log("Agent already awake (session active).")

                while not self._stop_flag.is_set():
                    try:
                        f32, pcm = self.audio_stream.read_frame()
                    except queue.Empty:
                        continue
                    except Exception:
                        continue

                    # --- audio level for Siri-like animation ---
                    try:
                        rms = float(np.sqrt(np.mean(np.square(f32))))
                    except Exception:
                        rms = 0.0
                    rms = min(max(rms * 4.0, 0.0), 1.0)  # amplify + clamp
                    self.state.level = 0.85 * self.state.level + 0.15 * rms

                    seg = self.vad.push(pcm, f32)
                    self.state.vad_active = bool(getattr(self.vad, "active", False))

                    if seg is None:
                        continue

                    utt = seg.astype(np.float32)
                    text, _, _ = core.transcribe_numpy(utt)
                    if not text:
                        continue

                    # Update last heard for UI (voice)
                    self.state.last_text = text
                    self._log(f"[Heard] {text}")

                    # Session control
                    if self.state.mode == "idle":
                        cand, score = core.hotword_detect(text)
                        if cand and score >= self.state.hotword_threshold:
                            self._log(f"🟢 Session started with '{cand}'")
                            try:
                                self.planner.sleep.keep_awake()
                            except Exception:
                                pass
                            self.state.mode = "session"
                        continue

                    if self.state.mode == "session":
                        end_cand, end_score = core.endword_detect(text)
                        if end_cand and end_score >= self.state.hotword_threshold:
                            self._log(f"🔴 Session ended with '{end_cand}'")
                            try:
                                self.planner.sleep.allow_sleep()
                            except Exception:
                                pass
                            self.state.mode = "idle"
                            continue

                        # NLU + Planner (may produce multiple commands)
                        cmds = process_text_commands(text)
                        if not cmds:
                            continue

                        for partial in cmds:
                            full_cmd = {
                                "module": "speech_nlu",
                                "ts": iso_now(),
                                **partial,
                            }

                            # update 'last_nlu' for UI
                            self.state.last_nlu = {
                                "intent": full_cmd.get("intent"),
                                "normalized": full_cmd.get("normalized"),
                                "entities": full_cmd.get("entities", []),
                            }

                            try:
                                log_nlu(full_cmd)
                            except Exception as e:
                                self._log(f"(log) log_nlu error: {e}")

                            intent_info = full_cmd.get("intent") or {}
                            label = intent_info.get("label")
                            conf = intent_info.get("confidence")
                            self._log(
                                f"(nlu) intent={label} conf={conf} "
                                f"norm={json.dumps(full_cmd.get('normalized') or {}, ensure_ascii=False)}"
                            )

                            try:
                                self.planner.handle(full_cmd)
                            except Exception as e:
                                self._log(
                                    f"[ERROR] Planner error: {e}\n{traceback.format_exc()}"
                                )

            except Exception as e:
                self._log(f"[FATAL] Listener crashed: {e}\n{traceback.format_exc()}")
            finally:
                try:
                    if self.audio_stream:
                        self.audio_stream.stop()
                except Exception:
                    pass
                self.state.listening = False
                self._log("Audio stopped.")


# =========================
# Global app state for NiceGUI
# =========================
@dataclass
class AppState:
    logger: ThreadLog
    log_lines: List[str]
    listener: BackgroundListener
    siri_html: Optional[ui.html] = None
    log_box: Optional[ui.html] = None
    last_heard_box: Optional[ui.html] = None
    last_nlu_box: Optional[ui.html] = None
    latest_email_box: Optional[ui.html] = None
    latest_email: Optional[dict] = None
    latest_email_ts: float = 0.0
    mic_device_index: Optional[int] = None
    samplerate: int = getattr(core, "SAMPLE_RATE", 16000)
    start_button: Optional[ui.button] = None
    stop_button: Optional[ui.button] = None

    # Caches to avoid redundant UI updates (helps reduce websocket traffic)
    _cache_siri_html: str = ""
    _cache_log_text: str = ""
    _cache_last_heard: str = ""
    _cache_last_nlu: str = ""
    _cache_email_html: str = ""


# create shared logger and listener
LOGGER = ThreadLog()
attach_gui_logger(LOGGER)
LISTENER = BackgroundListener(
    samplerate=getattr(core, "SAMPLE_RATE", 16000),
    mic_index=getattr(core, "MIC_DEVICE_INDEX", None),
    logger=LOGGER,
)

STATE = AppState(
    logger=LOGGER,
    log_lines=[],
    listener=LISTENER,
    mic_device_index=getattr(core, "MIC_DEVICE_INDEX", None),
)


# =========================
# Gmail helper (throttled)
# =========================
def fetch_latest_email_if_needed() -> Optional[dict]:
    # only refresh at most once every 60 seconds
    now = time.time()
    if STATE.latest_email and (now - STATE.latest_email_ts) < 60:
        return STATE.latest_email

    try:
        service = get_gmail_service()
        if not service:
            return STATE.latest_email
    except Exception:
        return STATE.latest_email

    try:
        resp = service.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            maxResults=1,
        ).execute()

        msgs = resp.get("messages", [])
        if not msgs:
            return STATE.latest_email

        msg = service.users().messages().get(
            userId="me",
            id=msgs[0]["id"],
            format="metadata",
            metadataHeaders=["From", "Subject", "Date"],
        ).execute()

        headers = {
            h["name"]: h["value"]
            for h in msg.get("payload", {}).get("headers", [])
        }

        from_h = headers.get("From", "Unknown")
        subj = headers.get("Subject", "(no subject)")
        date = headers.get("Date", "")

        if "<" in from_h:
            from_h = from_h.split("<")[0].strip()
        if len(subj) > 80:
            subj = subj[:77] + "…"

        STATE.latest_email = {"from": from_h, "subject": subj, "date": date}
        STATE.latest_email_ts = now
        return STATE.latest_email
    except Exception:
        # swallow errors, keep old email if any
        return STATE.latest_email


# =========================
# CSS (matching your Streamlit UI)
# =========================
ui.add_css(
    """
    body {
        background-color: #F2F3EF;
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .hero {
        background: linear-gradient(135deg, #111827, #020817);
        border-radius: 18px;
        padding: 12px 16px 10px 16px;
        border: 1px solid rgba(148,163,253,0.25);
        box-shadow: 0 6px 24px rgba(15,23,42,0.6); 
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: .02em;
        color: #e5e7eb;
    }
    .hero-sub {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 0.3rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 11px;
        border-radius: 999px;
        border: 1px solid rgba(148,163,253,0.35);
        font-size: 0.72rem;
        color: #c7d2fe;
        margin-bottom: 0.35rem;
        background: #020817;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #22c55e;
        box-shadow: 0 0 10px #22c55e;
    }
    .card {
        background: #020817;
        border-radius: 16px;
        padding: 10px 12px;
        border: 1px solid #4b5563;
        box-shadow: 0 10px 35px rgba(15,23,42,0.9);
        color: #bfc5d2;
    }
    .section-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: .14em;
        color: blue;
        margin-bottom: 0.15rem;
    }
    .log-box {
        font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 0.72rem !important;
        height: 560px;
        overflow-y: auto;
        background: #020817;
        border-radius: 14px;
        padding: 9px 10px 7px 10px;
        border: 1px solid #4b5563;
        color: #e5e7eb;
        white-space: pre-wrap;
        scroll-behavior: smooth;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: .12em;
    }
    .metric-value {
        font-size: 0.9rem;
        color: #cdd1db;
        font-weight: 600;
    }
    .voice-wrap {
        display:flex;
        gap:4px;
        align-items:flex-end;
        height:60px;
        margin-top: 8px;
    }
    .voice-bar {
        width:5px;
        border-radius:6px;
        background:#64748b;
        box-shadow:0 0 12px rgba(34,197,94,0.25);
        transition:height 25ms linear;
    }
    .voice-hint {
        font-size:.72rem;
        color:#9ca3af;
        margin-top:3px;
    }
    """
)


# =========================
# Siri animation renderer
# =========================
def build_siri_html() -> str:
    is_running = STATE.listener.is_running()
    bars = 32

    if not is_running:
        heights = [10 for _ in range(bars)]
        hint = "Press Start, then say “hey agent”."
    else:
        level = STATE.listener.state.level
        mode = STATE.listener.state.mode
        vad = STATE.listener.state.vad_active

        base = max(0.02, min(1.0, level))
        t = time.time() * 8

        heights = []
        for i in range(bars):
            h = 8 + abs(np.sin((i * 0.4) + t)) * (55 * base)
            h += random.uniform(-2, 2)
            heights.append(int(max(4, h)))

        if mode == "idle":
            hint = "Listening… say “hey agent”."
        else:
            hint = "Speak — I’m awake."

    color = "#22c55e" if is_running and STATE.listener.state.mode == "session" and STATE.listener.state.vad_active else "#64748b"
    glow = "0.9" if is_running and STATE.listener.state.mode == "session" and STATE.listener.state.vad_active else "0.25"

    bars_html = "".join(
        f"<div class='voice-bar' style='height:{h}px;background:{color};box-shadow:0 0 12px rgba(34,197,94,{glow});'></div>"
        for h in heights
    )

    html = f"""
    <div class="voice-wrap">
        {bars_html}
    </div>
    <div class="voice-hint">{hint}</div>
    """
    return html


# =========================
# Log updater
# =========================
def update_logs_and_ui():
    """
    Slower updates: drain logs (every ~1s) and update log box + latest email.
    We avoid updating UI elements when content has not changed to reduce websocket traffic.
    """
    # Drain logs from background thread
    drained = STATE.logger.drain(200)
    if drained:
        STATE.log_lines.extend(drained)
        STATE.log_lines = STATE.log_lines[-500:]

    # Update log box (only when changed)
    if STATE.log_box is not None:
        text = "\n".join(STATE.log_lines) if STATE.log_lines else "Waiting for events… say 'hey agent' or send a command."
        if text != STATE._cache_log_text:
            STATE._cache_log_text = text

            # Re-create the log-box HTML with a stable id so we can scroll it client-side
            STATE.log_box.content = f"<div id='logbox' class='log-box'>{text}</div>"

            # Run JS in the browser to scroll the log box to the bottom.
            # Use smooth scrolling for nicer UX.
            # Wrap in try/except to avoid crashing if JS can't run for some reason.
            try:
                ui.run_javascript(
                    "const el = document.getElementById('logbox');"
                    "if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });"
                )
            except Exception:
                # safe fallback — if JS cannot be executed, ignore
                pass


    # Update latest email (but not too often)
    email = fetch_latest_email_if_needed()
    if STATE.latest_email_box is not None:
        if email:
            email_html = (
                f"<div class='card' style='font-size:0.78rem;line-height:1.5;'>"
                f"<b>From:</b> {email['from']}<br/>"
                f"<b>Subject:</b> {email['subject']}<br/>"
                f"<span style='font-size:0.7rem;color:#9ca3af;'>{email['date']}</span>"
                f"</div>"
            )
        else:
            email_html = (
                "<div class='card' style='font-size:0.78rem;'>"
                "No email loaded yet. Say \"show me my latest email\"."
                "</div>"
            )

        if email_html != STATE._cache_email_html:
            STATE._cache_email_html = email_html
            STATE.latest_email_box.content = email_html


def update_siri_and_small():
    """
    Faster, lightweight updates (every ~100ms): Siri animation + last heard + last NLU + session card updates.
    Keep these cheap to avoid heavy websocket payloads.
    """
    # Update Siri animation (only when different)
    if STATE.siri_html is not None:
        new_siri = build_siri_html()
        if new_siri != STATE._cache_siri_html:
            STATE._cache_siri_html = new_siri
            STATE.siri_html.content = new_siri

    # Update last heard (cheap)
    if STATE.last_heard_box is not None:
        last_heard = STATE.listener.state.last_text or "—"
        if last_heard != STATE._cache_last_heard:
            STATE._cache_last_heard = last_heard
            STATE.last_heard_box.content = f"<div class='card' style='font-size:0.8rem;'>{last_heard}</div>"

    # Update last NLU (cheap)
    if STATE.last_nlu_box is not None:
        nlu = STATE.listener.state.last_nlu
        if nlu:
            pretty = json.dumps(nlu, indent=2, ensure_ascii=False)
            nlu_html = f"<div class='card' style='font-size:0.78rem;'><pre>{pretty}</pre></div>"
        else:
            nlu_html = "<div class='card' style='font-size:0.8rem;'>Awaiting command…</div>"

        if nlu_html != STATE._cache_last_nlu:
            STATE._cache_last_nlu = nlu_html
            STATE.last_nlu_box.content = nlu_html

    # Update session card + buttons (cheap)
    # We keep this in the faster path so UI feels responsive
    # Find the session_card (closure) by recreating its content function is not necessary;
    # Instead, trigger the same update_session_card logic that was created in build_ui via a small helper.
    # We'll only toggle button enable/disable states here.
    listening_flag = STATE.listener.is_running()
    if STATE.start_button is not None and STATE.stop_button is not None:
        if listening_flag:
            try:
                STATE.start_button.disable()
                STATE.stop_button.enable()
            except Exception:
                pass
        else:
            try:
                STATE.start_button.enable()
                STATE.stop_button.disable()
            except Exception:
                pass


# =========================
# UI builders
# =========================
def build_ui():
    # ===== Outer centering container =====
    with ui.column().style('width:100%; align-items:center;'):
        # inner content column, max width
        with ui.column().style(
            'width:100%; max-width:1200px; display:flex; flex-direction:column; gap:16px;'
        ):
            # ---------- HERO ----------
            ui.html(
                """
                <div class="hero">
                  <div class="pill">
                    <div class="status-dot"></div>
                    Agentic OS &nbsp;·&nbsp; Ambient desktop agent
                  </div>
                  <div class="hero-title">Talk to your system. One hotword away.</div>
                  <div class="hero-sub">
                    Start with <b>"hey agent"</b>, then say what you need:
                    open apps, control browser, inspect processes, check mail, or search for tickets.
                  </div>
                </div>
                """,
                sanitize=False,
            )

            # ---------- MAIN ROW: LEFT + RIGHT ----------
            with ui.row().style(
                'width:100%; display:flex; align-items:flex-start; gap:12px;'
            ):
                # =============== LEFT COLUMN ===============
                with ui.column().style(
                    'flex:3; min-width:0; display:flex; flex-direction:column; gap:12px;'
                ):
                    # Siri animation
                    STATE.siri_html = ui.html(build_siri_html(), sanitize=False)
                    ui.space().style('height:4px;')

                    # Controls row
                    with ui.row().style(
                        'width:100%; display:flex; align-items:flex-start; gap:16px;'
                    ):
                        # --- Start / Stop ---
                        with ui.column().style(
                            'flex:1; display:flex; flex-direction:column; gap:8px;'
                        ):
                            ui.html('<div class="section-label">Listening</div>', sanitize=False)

                            def on_start():
                                try:
                                    if STATE.mic_device_index is not None:
                                        sd.default.device = (STATE.mic_device_index, None)
                                    sd.default.samplerate = STATE.samplerate
                                except Exception:
                                    pass
                                STATE.listener.start()
                                STATE.logger.put("Requested start.")

                            def on_stop():
                                STATE.listener.stop()
                                STATE.logger.put("Requested stop.")

                            STATE.start_button = (
                                ui.button('▶ START', on_click=on_start)
                                .props('unelevated color="positive"')
                                .style(
                                    'width:100%; font-weight:600; border-radius:999px;'
                                )
                            )

                            STATE.stop_button = (
                                ui.button('⏹ STOP', on_click=on_stop)
                                .props('unelevated color="negative"')
                                .style(
                                    'width:100%; font-weight:600; border-radius:999px;'
                                )
                            )

                            # initially stopped -> stop disabled
                            STATE.stop_button.disable()

                        # --- Mic + samplerate ---
                        with ui.column().style(
                            'flex:1; display:flex; flex-direction:column; gap:8px;'
                        ):
                            ui.html('<div class="section-label">Input</div>', sanitize=False)

                            devices = []
                            try:
                                for i, dev in enumerate(sd.query_devices()):
                                    if dev.get("max_input_channels", 0) > 0:
                                        devices.append((i, f"{i}: {dev['name']}"))
                            except Exception as e:
                                ui.label(f"Device query failed: {e}").style(
                                    'font-size:11px; color:#f87171;'
                                )

                            options = [label for _, label in devices]
                            index_by_label = {label: idx for idx, label in devices}
                            if devices:
                                current = (
                                    STATE.mic_device_index
                                    if STATE.mic_device_index is not None
                                    else devices[0][0]
                                )
                                default_label = next(
                                    (lbl for idx, lbl in devices if idx == current),
                                    options[0],
                                )

                                def on_mic_change(e):
                                    label = e.value
                                    idx = index_by_label.get(label, devices[0][0])
                                    STATE.mic_device_index = idx
                                    try:
                                        sd.default.device = (STATE.mic_device_index, None)
                                        sd.default.samplerate = STATE.samplerate
                                    except Exception:
                                        pass
                                    STATE.listener.mic_index = STATE.mic_device_index

                                ui.select(
                                    options=options,
                                    value=default_label,
                                    on_change=on_mic_change,
                                ).props('dense outlined').style(
                                    'width:100%; font-size:12px;'
                                )

                            ui.html(
                                f"""
                                <div class="metric-label">Samplerate</div>
                                <div class="metric-value">{STATE.samplerate} Hz</div>
                                """,
                                sanitize=False,
                            )

                        # --- Typed command ---
                        with ui.column().style(
                            'flex:1; max-width:420px; display:flex; flex-direction:column; gap:8px;'
                        ):
                            ui.html('<div class="section-label">Type a command</div>', sanitize=False)

                            cmd_input = ui.input(
                                placeholder='hey agent  →  show me my latest email',
                            ).props('dense').style('width:100%; font-size:12px;')

                            def on_send():
                                cmd_txt = (cmd_input.value or "").strip()
                                cmd_input.value = ""
                                if not cmd_txt:
                                    return

                                # --- GEMINI enhancement for typed input (optional, safe) ---
                                try:
                                    if "_GEMINI_FOR_TYPED" in globals() and _GEMINI_FOR_TYPED and enhance_transcript_sync is not None:
                                        try:
                                            enhanced = enhance_transcript_sync(cmd_txt)
                                            if enhanced and enhanced.strip() != cmd_txt.strip():
                                                ts_log = time.strftime("%H:%M:%S")
                                                STATE.log_lines.append(f"[{ts_log}] [Typed] {cmd_txt}")
                                                STATE.log_lines.append(f"[{ts_log}] [Gemini] {enhanced}")
                                                cmd_txt = enhanced
                                        except Exception as e:
                                            STATE.log_lines.append(f"[{time.strftime('%H:%M:%S')}] (gemini typed) error: {e}")
                                except Exception:
                                    # defensive: if globals or names not present, just continue with original text
                                    pass

                                ts = time.strftime("%H:%M:%S")
                                STATE.log_lines.append(f"[{ts}] [Heard] {cmd_txt}")

                                # --- Same session logic as voice path ---
                                if STATE.listener.state.mode == "idle":
                                    cand, score = core.hotword_detect(cmd_txt)
                                    thr = STATE.listener.state.hotword_threshold
                                    if cand and score >= thr:
                                        STATE.log_lines.append(
                                            f"[{ts}] 🟢 Session started with '{cand}'"
                                        )
                                        try:
                                            STATE.listener.planner.sleep.keep_awake()
                                        except Exception:
                                            pass
                                        STATE.listener.state.mode = "session"
                                    else:
                                        STATE.log_lines.append(
                                            f"[{ts}] (idle) No hotword detected."
                                        )
                                    # In idle, we only use this to wake the agent.
                                    # No commands are executed in this turn.
                                    return

                                else:
                                    # In an active session: check for end word first
                                    end_cand, end_score = core.endword_detect(cmd_txt)
                                    if (
                                        end_cand
                                        and end_score >= STATE.listener.state.hotword_threshold
                                    ):
                                        STATE.log_lines.append(
                                            f"[{ts}] 🔴 Session ended with '{end_cand}'"
                                        )
                                        try:
                                            STATE.listener.planner.sleep.allow_sleep()
                                        except Exception:
                                            pass
                                        STATE.listener.state.mode = "idle"
                                        return

                                # --- If we reach here, we’re in session and should handle the command ---
                                cmds = process_text_commands(cmd_txt)
                                if not cmds:
                                    return

                                for partial in cmds:
                                    full_cmd = {
                                        "module": "typed_nlu",
                                        "ts": iso_now(),
                                        **partial,
                                    }

                                    try:
                                        log_nlu(full_cmd)
                                    except Exception as e:
                                        STATE.log_lines.append(
                                            f"[{ts}] (log) log_nlu error: {e}"
                                        )

                                    intent_info = full_cmd.get("intent") or {}
                                    label = intent_info.get("label")
                                    conf = intent_info.get("confidence")

                                    STATE.log_lines.append(
                                        f"[{ts}] (nlu) intent={label} conf={conf} "
                                        f"norm={json.dumps(full_cmd.get('normalized') or {}, ensure_ascii=False)}"
                                    )

                                    # update "last NLU" box from typed path
                                    STATE.listener.state.last_nlu = {
                                        "intent": full_cmd.get("intent"),
                                        "normalized": full_cmd.get("normalized"),
                                        "entities": full_cmd.get("entities", []),
                                    }

                                    # --- Run planner in background thread so UI doesn't block ---
                                    def worker(cmd_local, ts_local):
                                        out = _StreamToThreadLog(
                                            STATE.logger, mirror=sys.__stdout__
                                        )
                                        err = _StreamToThreadLog(
                                            STATE.logger,
                                            prefix="[stderr] ",
                                            mirror=sys.__stderr__,
                                        )
                                        try:
                                            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                                                STATE.listener.planner.handle(cmd_local)
                                        except Exception as e:
                                            STATE.logger.put(
                                                f"[{ts_local}] [ERROR] Planner error: {e}\n{traceback.format_exc()}"
                                            )

                                    threading.Thread(
                                        target=worker,
                                        args=(full_cmd, ts),
                                        daemon=True,
                                    ).start()


                            # Pressing Enter triggers on_send()
                            cmd_input.on('keydown.enter', lambda e: on_send())

                            ui.button('SEND', on_click=on_send).props(
                                'unelevated color="primary"'
                            ).style(
                                'width:100%; font-weight:600; border-radius:999px;'
                            )

                    # Live console
                    ui.html('<div class="section-label">Live console</div>', sanitize=False)
                    STATE.log_box = ui.html(
                        "<div id='logbox' class='log-box'>Waiting for events… say 'hey agent' or send a command.</div>",
                        sanitize=False,
                    ).style('width:100%;')


                # =============== RIGHT COLUMN ===============
                with ui.column().style(
                    'flex:0.8; min-width:260px; display:flex; flex-direction:column; gap:12px; margin-top:-130px;'
                ):
                    # Session card
                    ui.html('<div class="section-label">Session</div>', sanitize=False)

                    def session_card_html() -> str:
                        mode = STATE.listener.state.mode
                        listening_flag = STATE.listener.is_running()
                        backend = 'Listening' if listening_flag else 'Stopped'
                        return f"""
                        <div class="card">
                          <div class="metric-label">Backend</div>
                          <div class="metric-value">{backend}</div>
                          <div class="metric-label" style="margin-top:6px;">Mode</div>
                          <div class="metric-value">{mode}</div>
                        </div>
                        """

                    session_card = ui.html(session_card_html(), sanitize=False)

                    # Last heard
                    ui.html('<div class="section-label">Last heard</div>', sanitize=False)
                    STATE.last_heard_box = ui.html(
                        "<div class='card' style='font-size:0.8rem;'>—</div>",
                        sanitize=False,
                    )

                    # Last NLU
                    ui.html('<div class="section-label">Last NLU</div>', sanitize=False)
                    STATE.last_nlu_box = ui.html(
                        "<div class='card' style='font-size:0.8rem;'>Awaiting command…</div>",
                        sanitize=False,
                    )

                    # Hints
                    ui.html('<div class="section-label">Hints</div>', sanitize=False)
                    ui.html(
                        """
                        <div class="card" style="font-size:0.75rem; line-height:1.6;">
                          • "hey agent" → "show me my latest email"<br/>
                          • "show me my latest 5 unread emails"<br/>
                          • "show me cheapest flights from delhi to bangalore"<br/>
                          • "open file", "create a file", "append 'note' to notes.txt"<br/>
                          • "which process makes it slow", "keep awake", "open task manager"
                        </div>
                        """,
                        sanitize=False,
                    )

                    # Latest email
                    ui.html('<div class="section-label">Latest email</div>', sanitize=False)
                    STATE.latest_email_box = ui.html(
                        '<div class="card" style="font-size:0.78rem;">No email loaded yet. Say "show me my latest email".</div>',
                        sanitize=False,
                    )

                    # Timer: keep session card + buttons in sync with backend
                    def update_session_card():
                        listening_flag = STATE.listener.is_running()
                        session_card.content = session_card_html()

                        if STATE.start_button is not None and STATE.stop_button is not None:
                            if listening_flag:
                                STATE.start_button.disable()
                                STATE.stop_button.enable()
                            else:
                                STATE.start_button.enable()
                                STATE.stop_button.disable()

                    ui.timer(0.3, callback=update_session_card)


# =========================
# Global timers for logs + animation + email
# =========================
# IMPORTANT: Use two timers to avoid flooding websocket with frequent large updates.
# - update_siri_and_small: ~100ms for animation + small UI pieces (low payload)
# - update_logs_and_ui: ~1s for logs and email (higher payload, less frequent)
ui.timer(0.1, callback=update_siri_and_small)
ui.timer(1.0, callback=update_logs_and_ui)


# =========================
# Start app
# =========================
if __name__ in {"__main__", "__mp_main__"}:
    build_ui()
    print("Loading Whisper + NLP models…")
    ui.run(
        title='Agentic OS',
        host='0.0.0.0',
        port=8080,
        reload=False,
    )

