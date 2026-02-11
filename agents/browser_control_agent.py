# agents/browser_control_agent.py

import shutil
import subprocess
import time
import re

import psutil

from utils import run_cmd, open_with, looks_like_url, speak


class BrowserControlAgent:
    """
    Controls an existing browser window (Firefox/Chrome/Chromium) via xdotool/wmctrl.

    Supports:
      - "new tab"
      - "close tab", "next tab", "previous tab"
      - "back", "forward"
      - "scroll up/down/top/bottom"
      - "focus address bar"
      - "open url <...>", "go to <...>"
      - "browser search <query>", "type in address bar <query>"
    """

    def _has_proc(self, names=("firefox", "google-chrome", "chromium", "chromium-browser")) -> bool:
        """Return True if any browser process is running."""
        try:
            for p in psutil.process_iter(attrs=["name"]):
                n = (p.info.get("name") or "").lower()
                if any(n.startswith(x) for x in names):
                    return True
        except Exception:
            pass
        return False

    # ------------ Window activation helpers ------------

    def _activate_with_xdotool(self) -> bool:
        if not shutil.which("xdotool"):
            return False
        # Try common WM_CLASS names
        for cls in ["firefox", "Google-chrome", "Chromium", "chromium"]:
            wid = run_cmd([
                "bash",
                "-lc",
                f"xdotool search --onlyvisible --class '{cls}' | head -n 1",
            ]).strip()
            if wid and wid.isdigit():
                run_cmd(["bash", "-lc", f"xdotool windowactivate --sync {wid}"])
                return True
        return False

    def _activate_with_wmctrl(self) -> bool:
        if not shutil.which("wmctrl"):
            return False
        # Try by class/name
        for cls in ["firefox", "Google-chrome", "Chromium"]:
            # -xa activates by class/name, if present
            rc = subprocess.call(
                ["wmctrl", "-xa", cls],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if rc == 0:
                return True
        return False

    def _activate_browser(self) -> bool:
        """Try to focus any existing browser window."""
        return self._activate_with_wmctrl() or self._activate_with_xdotool()

    def _launch_once(self) -> bool:
        """
        Launch a browser once (prefer Firefox). After launch, attempt to focus it.
        """
        ok = open_with(["firefox", "google-chrome", "chromium-browser"])
        if ok:
            # Give WM a moment to map the window, then focus it
            time.sleep(0.8)
            self._activate_browser()
        return ok

    # ------------ Keystrokes / typing ------------

    def _send_keys(self, keys: str) -> bool:
        """Send keystrokes using xdotool."""
        if not shutil.which("xdotool"):
            print("(browser) Need xdotool for keystrokes. Install with: sudo apt install xdotool")
            return False
        run_cmd(["bash", "-lc", f"xdotool key {keys}"])
        return True

    def _type_and_enter(self, text: str) -> bool:
        """Type arbitrary text in the active window and press Enter."""
        if not shutil.which("xdotool"):
            print("(browser) Need xdotool for typing. Install with: sudo apt install xdotool")
            return False
        safe = text.replace('"', '\\"')
        run_cmd([
            "bash",
            "-lc",
            f'xdotool type --delay 3 --clearmodifiers "{safe}"',
        ])
        self._send_keys("Return")
        return True

    def _focus_omnibox(self) -> None:
        """Focus browser address bar (omnibox) via Ctrl+L."""
        self._send_keys("ctrl+l")
        time.sleep(0.05)

    # ------------ New tab helpers ------------

    def _remote_new_tab(self, target: str | None = None) -> bool:
        """
        Use browser 'remote' flags so a running instance handles the request
        without spawning a new window.

        For Firefox: `firefox --new-tab <URL-or-about:blank>`
        """
        # Prefer Firefox
        if shutil.which("firefox"):
            args = ["firefox", "--new-tab"]
            args.append(target if target else "about:blank")
            try:
                subprocess.Popen(
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(0.2)
                self._activate_browser()
                return True
            except Exception:
                return False

        # Chrome/Chromium fallback
        for chrome in ["google-chrome", "chromium-browser", "chromium"]:
            if shutil.which(chrome):
                args = [chrome, "--new-tab"]
                args.append(target if target else "about:blank")
                try:
                    subprocess.Popen(
                        args,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    time.sleep(0.2)
                    self._activate_browser()
                    return True
                except Exception:
                    pass

        return False

    # ------------ Public entry ------------

    def control(self, cmd: dict) -> None:
        """
        Handle browser control commands from Planner.

        Expects:
          cmd["original_text"]: the spoken utterance
          cmd["normalized"]:    NLU-normalized fields (goto_target, search_query, ...)
        """
        text = (cmd.get("original_text") or "")
        norm = (cmd.get("normalized") or {}) or {}
        t = text.lower().strip()

        # 1) Try to activate an existing browser window; don't relaunch unless needed
        activated = self._activate_browser()
        has_proc = self._has_proc()

        # ------------ "new tab" ------------
        if "new tab" in t or norm.get("browser_action") == "new_tab":
            if has_proc:
                if not self._remote_new_tab():
                    print("(browser) Remote new-tab failed; ensure Firefox or Chrome is installed.")
            else:
                # No running browser — launch once, then open a new tab
                if self._launch_once():
                    self._remote_new_tab()
            return

        # ------------ Focus address bar ------------
        if "focus address bar" in t or "address bar" in t:
            if not activated and not has_proc:
                self._launch_once()
            self._focus_omnibox()
            return

        # ------------ Go to URL / website ------------
        goto = norm.get("goto_target")
        if goto:
            # Defensive normalization of goto_target (safe fallback)
            g_raw = str(goto or "").strip()
            g = g_raw

            # normalize spoken "dot"/"period" -> '.' and remove stray punctuation/extra spaces
            g = re.sub(r'\b(dot|period)\b', '.', g, flags=re.IGNORECASE)
            g = re.sub(r'\s*\.\s*', '.', g)  # normalize spaces around dots
            g = re.sub(r'[,\s/]+$', '', g).strip()  # rstrip stray separators
            g = re.sub(r'^\s*[\.]+', '', g).strip()  # lstrip stray dots
            g = re.sub(r'\s+', ' ', g).strip()

            # If it contains spaces and no dot, try collapsing short alnum tokens into a compact hostname
            if (' ' in g or any(c.isupper() for c in g)) and '.' not in g:
                tokens = re.findall(r'[A-Za-z0-9\-]+', g)
                if 1 < len(tokens) <= 3 and all(len(tok) <= 20 for tok in tokens):
                    collapsed = ''.join(tokens).lower()
                    # conservative length check
                    if re.fullmatch(r'[a-z0-9\-]{2,60}', collapsed):
                        g = collapsed

            # If still no dot, and it's a single compact token, append .com (conservative)
            if '.' not in g:
                simple_token = re.fullmatch(r'[A-Za-z0-9\-]{2,30}', g)
                if simple_token:
                    # If user explicitly said com/org/net in raw text, respect that by appending .com
                    if re.search(r'\b(com|org|net|io|co|in)\b', g_raw, re.IGNORECASE) or True:
                        g = (g.lower() + '.com').strip('.')

            # Final trim
            g = g.strip().strip('"').strip("'")

            # Prepare goto value to use below
            goto_final = g

            if not activated and not has_proc:
                self._launch_once()

            # Ensure URL looks valid to our heuristic; if not, prefix https://
            if not looks_like_url(goto_final):
                goto_final = "https://" + goto_final

            # Prefer remote new-tab when browser is running
            if has_proc and self._remote_new_tab(goto_final):
                return

            # Fallback to typing in omnibox
            if not activated:
                # Try to activate again (in case we just launched)
                self._activate_browser()
            self._focus_omnibox()
            self._type_and_enter(goto_final)
            return

        # ------------ In-page navigation ------------
        nav_keys = None
        if "close tab" in t or norm.get("browser_action") == "close_tab":
            nav_keys = "ctrl+w"
        elif "next tab" in t:
            nav_keys = "ctrl+Tab"
        elif "previous tab" in t or "prev tab" in t:
            nav_keys = "ctrl+shift+Tab"
        elif "back" in t:
            nav_keys = "Alt+Left"
        elif "forward" in t:
            nav_keys = "Alt+Right"
        elif "scroll to top" in t:
            nav_keys = "Home"
        elif "scroll to bottom" in t:
            nav_keys = "End"
        elif "scroll down" in t:
            nav_keys = "Page_Down"
        elif "scroll up" in t:
            nav_keys = "Page_Up"

        if nav_keys:
            # Ensure a browser window is focused; try activation if a browser process exists
            if not activated:
                if has_proc:
                    # attempt to activate window now (maybe WM didn't respond earlier)
                    activated = self._activate_browser()
                    time.sleep(0.05)
                else:
                    # No browser at all — try to launch one once (then activate)
                    if self._launch_once():
                        activated = True
                        has_proc = True

            if not activated:
                # still not focused -> warn user
                if has_proc:
                    print(
                        "(browser) Browser running but could not focus it. "
                        "On Wayland, keystrokes may not work; try XWayland or run Firefox with MOZ_ENABLE_WAYLAND=0."
                    )
                else:
                    print("(browser) No browser window found. Say 'open firefox' first.")
                return

            # Now we have browser focused — send keystrokes
            self._send_keys(nav_keys)
            return

        # ------------ Browser search (omnibox) ------------
        if any(p in t for p in ["browser search", "type in address bar"]):
            q = norm.get("search_query")
            if not q:
                q = text
                for w in ["browser search", "type in address bar", "search for", "search"]:
                    q = q.replace(w, "")
                q = q.strip()

            if not activated and not has_proc:
                self._launch_once()
                self._activate_browser()

            self._focus_omnibox()
            self._type_and_enter(q or " ")
            return

        print("(browser) No control action matched.")

