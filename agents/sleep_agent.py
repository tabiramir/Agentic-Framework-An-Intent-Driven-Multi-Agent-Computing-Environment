# agents/sleep_agent.py

import subprocess

from utils import which, speak


class SleepAgent:
    def __init__(self):
        self.inhibitor = None

    def keep_awake(self):
        if self.inhibitor and self.inhibitor.poll() is None:
            print("(sleep) Already preventing sleep.")
            return
        exe = which(["systemd-inhibit"])
        if not exe:
            print("(sleep) systemd-inhibit not found. Install systemd or use caffeine.")
            return
        try:
            self.inhibitor = subprocess.Popen(
                [
                    exe,
                    "--what=handle-lid-switch:sleep:idle",
                    "--why=AgenticOS keep awake",
                    "sleep",
                    "infinity",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("(sleep) Sleep prevention started (systemd-inhibit).")
        except Exception as e:
            print(f"(sleep) Failed to start inhibitor: {e}")

    def allow_sleep(self):
        if self.inhibitor and self.inhibitor.poll() is None:
            try:
                self.inhibitor.terminate()
                self.inhibitor = None
                print("(sleep) Sleep prevention stopped.")
            except Exception as e:
                print(f"(sleep) Failed to stop inhibitor: {e}")
        else:
            print("(sleep) No active inhibitor.")

    def handle(self, cmd):
        t = (cmd.get("original_text") or "").lower()
        if any(
            p in t
            for p in [
                "allow sleep",
                "stop preventing sleep",
                "let it sleep",
                "disable keep awake",
                "stop keep awake",
            ]
        ):
            self.allow_sleep()
            return
        if any(
            p in t
            for p in [
                "don't sleep",
                "dont sleep",
                "keep awake",
                "keep running",
                "prevent sleep",
                "caffeinate",
                "stay awake",
                "keep system awake",
                "no sleep",
            ]
        ):
            self.keep_awake()
            return
        self.keep_awake()

