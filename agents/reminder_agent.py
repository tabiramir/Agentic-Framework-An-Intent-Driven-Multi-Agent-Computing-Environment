# agents/reminder_agent.py

import re
import threading
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import dateparser
from dateutil import parser as dtparse
from bson.objectid import ObjectId

from config import IST, MONGO_COLLECTION_REMINDERS
from utils import speak, log_agent, iso_now, gui_log
from db import get_db


class ReminderAgent:
    def __init__(self):
        # We keep timers mainly so they stay referenced and don't get GC'ed.
        # Each entry is a dict: {"timer": threading.Timer, "reminder_id": str | None}
        self.timers: list[Dict[str, Any]] = []

    # --------- Time parsing helpers ---------

    def _parse_when(self, cmd: Dict[str, Any]):
        """
        Decide when the reminder should fire.

        Priority:
        1. Explicit "in/after/for X {seconds/minutes/hours/days}" → relative from now
        2. NLU normalized datetime, but ONLY if it's in the future
        3. Fallback: dateparser over the whole sentence
        All in IST.
        """
        text = (cmd.get("original_text") or "").lower().strip()
        norm = (cmd.get("normalized") or {}) or {}
        now = datetime.now(IST)

        # 1) Relative offset: "in 2 minutes", "for 5 min"
        m = re.search(
            r'\b(?:in|after|for)\s+(\d+)\s*'
            r'(second|seconds|sec|secs|s|minute|minutes|min|mins|m|hour|hours|hr|hrs|day|days|d)\b',
            text,
        )
        if m:
            amount = int(m.group(1))
            unit = m.group(2)

            if unit.startswith(("second", "sec", "s")):
                return now + timedelta(seconds=amount)
            if unit.startswith(("minute", "min", "m")):
                return now + timedelta(minutes=amount)
            if unit.startswith(("hour", "hr", "h")):
                return now + timedelta(hours=amount)
            if unit.startswith("day") or unit == "d":
                return now + timedelta(days=amount)

        # 2) Use normalized datetime ONLY if it's clearly in the future
        dt_val = norm.get("datetime")
        if dt_val:
            try:
                dt = dtparse.isoparse(dt_val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=IST)
                else:
                    dt = dt.astimezone(IST)

                if dt > now + timedelta(seconds=2):
                    return dt
            except Exception:
                pass

        # 3) Fallback: natural language parse of full text
        try:
            dt = dateparser.parse(
                text,
                settings={
                    "TIMEZONE": "Asia/Kolkata",
                    "RETURN_AS_TIMEZONE_AWARE": True,
                    "PREFER_DATES_FROM": "future",
                },
            )
            if dt:
                return dt.astimezone(IST)
        except Exception as e:
            print(f"(reminder) dateparser error: {e}")

        return None

    # --------- MongoDB helpers ---------

    def _save_reminder(
        self, text: str, when_dt: datetime, cmd: Dict[str, Any]
    ) -> Optional[str]:
        """Insert a reminder document into MongoDB. Return reminder_id (string) or None."""
        db = get_db()
        if db is None:
            return None
        try:
            doc = {
                "text": text,
                "when": when_dt.isoformat(),
                "status": "scheduled",
                "created_at": iso_now(),
                "last_updated": iso_now(),
                "source_module": cmd.get("module"),
                "original_cmd": cmd,
            }
            res = db[MONGO_COLLECTION_REMINDERS].insert_one(doc)
            return str(res.inserted_id)
        except Exception as e:
            print(f"(reminder) mongo insert error: {e}")
            return None

    def _mark_fired(self, reminder_id: Optional[str]) -> None:
        """Mark a reminder as fired in MongoDB."""
        if not reminder_id:
            return
        db = get_db()
        if db is None:
            return
        try:
            db[MONGO_COLLECTION_REMINDERS].update_one(
                {"_id": ObjectId(reminder_id)},
                {
                    "$set": {
                        "status": "fired",
                        "last_updated": iso_now(),
                        "fired_at": iso_now(),
                    }
                },
            )
        except Exception as e:
            print(f"(reminder) mongo update error: {e}")

    # --------- Public API ---------

    def create(self, cmd: Dict[str, Any]) -> None:
        text = (cmd.get("original_text") or "").strip() or "reminder"
        when_dt = self._parse_when(cmd)

        if not when_dt:
            print("(reminder) Missing/unclear time for reminder.")
            speak("I couldn't understand when to set the reminder.")
            return

        now = datetime.now(IST)
        delay = (when_dt - now).total_seconds()

        if delay <= 0:
            print("⏰ Reminder time already passed or is now, firing immediately.")
            # Save as fired immediately
            reminder_id = self._save_reminder(text, when_dt, cmd)
            self._trigger(text, reminder_id)
            return

        print(f"Reminder set for {when_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}: {text}")
        speak(f"Reminder scheduled for {when_dt.strftime('%Y-%m-%d %H:%M:%S')} IST")

        # Save in DB as scheduled
        reminder_id = self._save_reminder(text, when_dt, cmd)

        # Start timer that will fire this reminder
        t = threading.Timer(delay, self._trigger, args=[text, reminder_id])
        t.daemon = True
        t.start()
        self.timers.append({"timer": t, "reminder_id": reminder_id})

        try:
            log_agent(
                {
                    "agent": "reminder",
                    "event": "scheduled",
                    "text": text,
                    "when": when_dt.isoformat(),
                    "reminder_id": reminder_id,
                    "ts": iso_now(),
                }
            )
        except Exception:
            pass

    def _trigger(self, text: str, reminder_id: Optional[str] = None) -> None:
        phrases = [
            "Reminder activated!",
            "Hey, your reminder is up.",
            "Time’s up — check your task!",
            "Your reminder just went off!",
        ]
        msg = random.choice(phrases)

        gui_log(f"🔔 {msg} ({text})")
        speak(msg)

        # Update DB status
        self._mark_fired(reminder_id)

        try:
            log_agent(
                {
                    "agent": "reminder",
                    "event": "fired",
                    "text": text,
                    "reminder_id": reminder_id,
                    "ts": iso_now(),
                }
            )
        except Exception:
            pass

