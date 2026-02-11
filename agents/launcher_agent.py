# agents/launcher_agent.py

from pathlib import Path

from utils import open_with, speak


class LauncherAgent:
    ALLOWLIST = {
        "firefox": ["firefox"],
        "chrome": ["google-chrome", "chromium-browser"],
        "terminal": ["gnome-terminal", "x-terminal-emulator", "konsole", "xfce4-terminal"],
        "vscode": ["code"],
        "spotify": ["spotify"],
        "settings": ["gnome-control-center", "unity-control-center"],
        "calculator": [
        "gnome-calculator",
        "kcalc",
        "galculator",
        "xcalc",
        "mate-calc",
        ],
    }

    def launch(self, cmd):
        norm = (cmd.get("normalized") or {}) or {}
        app = norm.get("application")
        text = (cmd.get("original_text") or "").lower()

        if not app:
            for k in self.ALLOWLIST.keys():
                if k in text:
                    app = k
                    break

        if "calculator" in text and not app:
            app = "calculator"

        if app and app in self.ALLOWLIST:
            if open_with(self.ALLOWLIST[app]):
                speak(f"Opened {app}")
                return

        if "music" in text:
            open_with(["nautilus", "xdg-open"], [str(Path.home() / "Music")])
            speak("Opened Music folder.")
            return

        if "downloads" in text:
            open_with(["nautilus", "xdg-open"], [str(Path.home() / "Downloads")])
            speak("Opened Downloads folder.")
            return

        print(f"(launcher) Could not launch {app}")
        speak(f"Could not launch {app or 'that'}")

