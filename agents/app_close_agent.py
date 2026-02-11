# agents/app_close_agent.py

import os
import re
import shutil
import subprocess

import psutil

from utils import run_cmd, speak


class AppCloseAgent:
    # Prevent killing system-critical processes
    PROTECTED = [
        "systemd",
        "init",
        "dbus",
        "xorg",
        "xwayland",
        "wayland",
        "gnome-shell",
        "plasmashell",
        "kdeinit",
        "pipewire",
        "pulseaudio",
        "login",
        "bash",
        "zsh",
    ]

    def graceful_close_windows(self, appname: str):
        if shutil.which("xdotool"):
            try:
                run_cmd(
                    [
                        "bash",
                        "-lc",
                        f"xdotool search --onlyvisible --class '{appname}'"
                        " | xargs -I {} xdotool windowclose {}",
                    ]
                )
                print(f"(close) graceful windowclose for {appname}")
            except Exception:
                pass

    def kill_by_name(self, name: str):
        if any(p == name.lower() for p in self.PROTECTED):
            print(f"(close) Protected process '{name}' skipped")
            return False
        try:
            subprocess.Popen(
                ["pkill", "-f", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"(close) pkill by name: {name}")
            return True
        except Exception as e:
            print(f"(close) pkill failed: {e}")
            return False

    def close_all(self, keyword=""):
        print(f"(close) Close all matching '{keyword}'")
        cnt = 0
        for p in psutil.process_iter(attrs=["pid", "name"]):
            try:
                pname = (p.info.get("name") or "").lower()
                if keyword:
                    if keyword in pname and pname not in self.PROTECTED:
                        os.kill(p.info["pid"], 9)
                        cnt += 1
                else:
                    if pname not in self.PROTECTED and any(
                        app in pname for app in ["firefox", "chrome", "code", "vlc", "spotify"]
                    ):
                        os.kill(p.info["pid"], 9)
                        cnt += 1
            except Exception:
                pass
        return cnt

    def handle(self, cmd):
        text = (cmd.get("original_text") or "").lower().strip()

        # Kill by specific PID
        m = re.search(r"kill process (\d+)", text)
        if m:
            try:
                os.kill(int(m.group(1)), 9)
                speak(f"Killed process {m.group(1)}")
            except Exception:
                speak("Could not kill that process")
            return

        # Close everything
        if "close everything" in text or "close all" in text:
            killed = self.close_all()
            speak(f"Closed {killed} apps")
            return

        # Extract app name from sentence after "close" / "quit" / "stop"
        m2 = re.search(r"(?:close|quit|exit|stop)\s+(.+)$", text)
        if m2:
            name = m2.group(1).strip().lower()

            self.graceful_close_windows(name)
            ok = self.kill_by_name(name)
            speak(f"Closed {name}" if ok else f"Couldn't close {name}")
            return

        speak("Not sure what to close.")

