# agents/file_manager_agent.py

import re
import shutil
import subprocess
from pathlib import Path

from utils import expand_dir_keyword, speak, run_cmd


class FileManagerAgent:
    """
    Conversational file operations with slot-filling:
      - "create a file" → ask filename → ask location → create
      - "open file" → ask location → list files → ask filename → open
      - "append '...'" → requires an opened/selected file → append
      - "close file" → clears current file context and closes the editor app (best-effort)
      - "delete file" → ask location → list files → ask filename → delete
    """

    FILE_MANAGERS = ["nautilus", "nemo", "thunar", "dolphin", "pcmanfm", "xdg-open"]

    def __init__(self):
        self.state = {
            "mode": None,          # "create" | "open" | "edit" | "delete"
            "await": None,         # "filename" | "location" | None
            "dir_kw": None,        # "downloads" | ...
            "filename": None,      # "notes.txt"
            "current_file": None,  # Path
        }
        # which editor app we used to open the current file (e.g. gedit, gnome-text-editor)
        self.current_editor_program: str | None = None

    def in_dialog(self) -> bool:
        return self.state["mode"] is not None or self.state["await"] is not None

    # ---------- helpers ----------

    def _open_file_manager(self, path: Path = None):
        path = path or Path.home()
        for cmd in self.FILE_MANAGERS:
            if shutil.which(cmd):
                try:
                    subprocess.Popen([cmd, str(path)])
                    print(f"(file) 📂 Opened file manager at {path}")
                    return True
                except Exception:
                    continue
        print(f"(file) ❌ No supported file manager found for {path}")
        return False

    def _list_files(self, folder: Path):
        try:
            if not folder.exists():
                return []
            items = [p for p in folder.iterdir() if p.is_file()]
            items.sort(key=lambda p: p.name.lower())
            return items
        except Exception as e:
            print(f"(file) list error: {e}")
            return []

    def _ensure_ext(self, name: str) -> str:
        name = name.strip().strip('\'"')
        if "." not in name:
            name += ".txt"
        return name

    def _extract_append_content(self, user_text: str):
        m = re.search(r'"([^"]+)"|\'([^\']+)\'', user_text)
        if m:
            return (m.group(1) or m.group(2)).strip()
        m2 = re.search(r"\bappend\b(.*)$", user_text, re.IGNORECASE)
        if not m2:
            return None
        content = m2.group(1).strip()
        return content or None

    def _parse_dir_from_text(self, t: str):
        m = re.search(
            r"\b(?:in|to|into|at|under|inside)\s+"
            r"(home|downloads?|documents?|desktop|pictures?|music|videos?)\b",
            t,
            re.I,
        )
        if m:
            return m.group(1).lower().rstrip("s") + "s"
        for dk in ["downloads", "documents", "desktop", "pictures", "music", "videos", "home"]:
            if re.search(rf"\b{dk}\b", t, re.I):
                return dk
        return None

    def _get_folder(self) -> Path:
        return expand_dir_keyword(self.state["dir_kw"])

    def _create_or_touch(self, folder: Path, name: str):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            p = folder / name
            if not p.exists():
                p.touch()
                print(f"(file) Created: {p}")
            else:
                print(f"(file) Exists: {p}")
            return p
        except Exception as e:
            print(f"(file) ❌ Create failed: {e}")
            return None

    def _append_text(self, file_path: Path, text: str) -> bool:
        try:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(text + ("" if text.endswith("\n") else "\n"))
            print(f"(file) Appended to {file_path}")
            return True
        except Exception as e:
            print(f"(file) ❌ Append failed: {e}")
            return False

    def _open_with_default(self, file_path: Path) -> bool:
        """
        Open file with a GUI editor and remember which program we used,
        so that 'close file' can kill that process.
        """
        # Common GUI text editors on Ubuntu and other desktops
        editors = [
            "gnome-text-editor",  # default "Text Editor" on newer Ubuntu
            "gedit",
            "xed",
            "pluma",
            "kate",
            "leafpad",
            "mousepad",
        ]

        # Try editors directly first
        for exe in editors:
            if shutil.which(exe):
                try:
                    subprocess.Popen([exe, str(file_path)])
                    print(f"(file) Opened file {file_path} with {exe}")
                    self.current_editor_program = exe
                    return True
                except Exception as e:
                    print(f"(file) ❌ Failed to open with {exe}: {e}")
                    continue

        # Fallback to xdg-open (we will not pkill xdg-open itself later)
        try:
            subprocess.Popen(["xdg-open", str(file_path)])
            print(f"(file) Opened file {file_path} with xdg-open")
            # we don't know the real editor app -> leave None,
            # and in close we'll try generic pkill on common editors
            self.current_editor_program = None
            return True
        except Exception as e:
            print(f"(file) ❌ Open file failed: {e}")
            self.current_editor_program = None
            return False

    def _clean_name(self, s: str) -> str:
        s = s.strip().strip('\'"')
        s = re.sub(r"\.+$", "", s)
        return s

    # ---------- dialog starts ----------

    def _start_create(self):
        self.state.update({"mode": "create", "await": "filename", "dir_kw": None, "filename": None})
        speak("What should be the file name?")

    def _start_open(self):
        self.state.update({"mode": "open", "await": "location", "dir_kw": None, "filename": None})
        speak("Where should I look? You can say for example, in Downloads folder.")

    def _start_delete(self):
        self.state.update({"mode": "delete", "await": "location", "dir_kw": None, "filename": None})
        speak("Which location? Say like, in Documents or in Downloads.")

    def _close_current(self):
        """
        Close the current file from the agent's perspective AND
        try to close the editor process we used to open it.
        """
        if self.state["current_file"]:
            fname = self.state["current_file"].name
            editor = self.current_editor_program

            # Clear state first
            self.state["current_file"] = None
            self.current_editor_program = None

            speak(f"Closed {fname}.")

            # If we know which editor we used, pkill that
            if editor:
                try:
                    out = run_cmd(["pkill", "-f", editor])
                    print(f"(file) pkill '{editor}' output: {out.strip()}")
                except Exception as e:
                    print(f"(file) error trying to pkill {editor}: {e}")
            else:
                # Fallback: try to close any common text-editor processes
                candidates = [
                    "gnome-text-editor",
                    "gedit",
                    "xed",
                    "pluma",
                    "kate",
                    "leafpad",
                    "mousepad",
                ]
                for exe in candidates:
                    try:
                        out = run_cmd(["pkill", "-f", exe])
                        if out.strip():
                            print(f"(file) generic pkill '{exe}' output: {out.strip()}")
                    except Exception as e:
                        print(f"(file) error trying to pkill {exe}: {e}")
        else:
            speak("No file is open right now.")

    # ---------- top-level entry ----------

    def manage(self, cmd):
        text = (cmd.get("original_text") or "").strip()
        t = text.lower()

        # quick append when a file is already open
        if "append" in t and self.state.get("current_file") and not self.in_dialog():
            content = self._extract_append_content(text)
            if not content:
                speak("I didn't catch what to append. Say for example: append hello world.")
                return
            if self._append_text(self.state["current_file"], content):
                speak("Added to the file.")
            else:
                speak("I couldn't write to the file.")
            return

        # close file
        if re.search(r"\bclose\b.*\bfile\b", t):
            self._close_current()
            self.state.update({"mode": None, "await": None})
            return

        if re.search(r"\b(open|show)\s+(?:the\s+)?(file manager|files app|files)\b", t):
            if self._open_file_manager(Path.home()):
                speak("Opened file manager.")
            else:
                speak("I couldn't open the file manager.")
            self.state.update({"mode": None, "await": None})
            return

        m_folder = re.search(
            r"\bopen\s+(?:the\s+)?(downloads?|documents?|desktop|pictures?|music|videos?|home)\s*(folder)?\b",
            t,
        )
        if m_folder:
            dir_kw = m_folder.group(1)
            folder = expand_dir_keyword(dir_kw)
            if self._open_file_manager(folder):
                name = folder.name if folder.name else "folder"
                speak(f"Opened {name} folder.")
            else:
                speak("I couldn't open that folder.")
            self.state.update({"mode": None, "await": None})
            return

        if self.in_dialog():
            self._continue_dialog(text)
            return

        if re.search(r"\bcreate (?:a )?file\b", t):
            self._start_create()
            return

        if re.search(r"\bopen\s+(?:a\s+)?file\b(?!\s*manager)", t):
            self._start_open()
            return

        if re.search(r"\bdelete (?:a )?file\b", t) or re.search(r"\bdelete file\b", t):
            self._start_delete()
            return

        if re.search(r"\bedit (?:a )?file\b", t):
            if self.state["current_file"]:
                speak("You can say append hello world to add text, then say close file when done.")
            else:
                speak("Open a file first. Say: open file.")
            return

        speak(
            "Say open file manager, open downloads folder, create a file, "
            "open file, append hello world, close file, or delete file."
        )

    # ---------- dialog continuation ----------

    def _continue_dialog(self, user_text: str):
        t = user_text.lower()

        if "append" in t and self.state["current_file"]:
            content = self._extract_append_content(user_text)
            if not content:
                speak("I didn't catch what to append. Say for example: append hello world.")
                return
            if self._append_text(self.state["current_file"], content):
                speak("Added to the file.")
            else:
                speak("I couldn't write to the file.")
            return

        if re.search(r"\bclose\b.*\bfile\b", t):
            self._close_current()
            self.state.update({"mode": None, "await": None})
            return

        if self.state["await"] == "filename":
            m_q = re.search(r'"([^"]+)"|\'([^\']+)\'', user_text)
            name = (m_q.group(1) or m_q.group(2)) if m_q else user_text.strip()
            name = self._clean_name(name)

            if not name:
                speak("I didn’t catch the name. Please repeat the file name.")
                return

            if self.state["mode"] == "create":
                name = self._ensure_ext(name)
                self.state["filename"] = name
                if not self.state["dir_kw"]:
                    self.state["await"] = "location"
                    speak("Where should I save it?")
                    return

                folder = self._get_folder()
                p = self._create_or_touch(folder, self.state["filename"])
                if p:
                    self.state["current_file"] = p
                    self._open_file_manager(folder)
                    speak(f"Saved {p.name} in {folder.name}.")
                else:
                    speak("I couldn't create the file.")
                self.state.update({"mode": None, "await": None})
                return

            self.state["filename"] = name

        if self.state["await"] == "location":
            dir_kw = self._parse_dir_from_text(t)
            if not dir_kw:
                speak("Please tell me a location, for example: in Downloads folder.")
                return
            self.state["dir_kw"] = dir_kw
            folder = self._get_folder()

            if self.state["mode"] == "create":
                if not self.state["filename"]:
                    self.state["await"] = "filename"
                    speak("What should be the file name?")
                    return
                p = self._create_or_touch(folder, self.state["filename"])
                if p:
                    self.state["current_file"] = p
                    self._open_file_manager(folder)
                    speak(f"Saved {p.name} in {folder.name}.")
                else:
                    speak("I couldn't create the file.")
                self.state.update({"mode": None, "await": None})
                return

            if self.state["mode"] == "open":
                items = self._list_files(folder)
                if not items:
                    speak(f"I didn’t find files in {folder.name}.")
                    self.state.update({"mode": None, "await": None})
                    return
                names = [p.name for p in items][:8]
                print("(file) Files in", folder, ":\n - " + "\n - ".join([p.name for p in items]))
                speak("I found: " + ", ".join(names) + ". Which file should I open?")
                self.state["await"] = "filename"
                return

            if self.state["mode"] == "delete":
                items = self._list_files(folder)
                if not items:
                    speak(f"No files found in {folder.name}.")
                    self.state.update({"mode": None, "await": None})
                    return
                names = [p.name for p in items][:8]
                print("(file) Files in", folder, ":\n - " + "\n - ".join([p.name for p in items]))
                speak("Which file should I delete? For example, say the file name.")
                self.state["await"] = "filename"
                return

        if self.state["mode"] == "open" and self.state["await"] == "filename":
            folder = self._get_folder()
            if not folder:
                speak("Please tell me a location first.")
                self.state["await"] = "location"
                return

            m_q = re.search(r'"([^"]+)"|\'([^\']+)\'', user_text)
            name = (m_q.group(1) or m_q.group(2)) if m_q else user_text.strip()

            candidates = self._list_files(folder)
            if not candidates:
                speak(f"No files found in {folder.name}.")
                self.state.update({"mode": None, "await": None})
                return

            pick = None
            for p in candidates:
                if p.name.lower() == name.lower():
                    pick = p
                    break
            if not pick:
                for p in candidates:
                    if p.name.lower().startswith(name.lower()) or name.lower() in p.name.lower():
                        pick = p
                        break
            if not pick:
                speak("I couldn't find that file. Please say the name again.")
                return

            self.state["current_file"] = pick
            self._open_with_default(pick)
            speak(
                f"Opened {pick.name}. You can say append hello world to add text, "
                "then say close file when done."
            )
            self.state.update({"mode": None, "await": None})
            return

        if self.state["mode"] == "delete" and self.state["await"] == "filename":
            folder = self._get_folder()
            m_q = re.search(r'"([^"]+)"|\'([^\']+)\'', user_text)
            name = (m_q.group(1) or m_q.group(2)) if m_q else user_text.strip()
            target = folder / name

            if not target.exists() and "." not in name:
                if (folder / (name + ".txt")).exists():
                    target = folder / (name + ".txt")
            if not target.exists():
                cands = [p for p in self._list_files(folder) if name.lower() in p.name.lower()]
                if cands:
                    target = cands[0]
            if not target.exists():
                speak("I couldn't find that file to delete. Please say the name again.")
                return
            try:
                target.unlink()
                speak(f"Deleted {target.name}.")
            except Exception as e:
                print(f"(file) delete error: {e}")
                speak("I couldn't delete it.")
            self.state.update({"mode": None, "await": None})
            return

        speak("I didn’t catch that. Please repeat.")

