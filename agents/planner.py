# agents/planner.py

from typing import Dict, Any

from agents.reminder_agent import ReminderAgent
from agents.web_agent import WebAgent
from agents.launcher_agent import LauncherAgent
from agents.app_close_agent import AppCloseAgent
from agents.file_manager_agent import FileManagerAgent
from agents.process_manager_agent import ProcessManagerAgent
from agents.sleep_agent import SleepAgent
from agents.mail_agent import MailAgent
from agents.booking_agent import BookingAgent
from agents.browser_control_agent import BrowserControlAgent

# Map NLU intent prefixes → which agent should handle it
INTENT_AGENT_MAP = {
    "reminder": "reminder",
    "web": "web",
    "app": "launcher",
    "file": "file",
    "process": "process",
    "sleep": "sleep",
    "browser": "browser",
    "close": "close",
    "booking": "booking",
    "mail": "mail",
    "music": "launcher",
}


class Planner:
    """
    Central router that takes a `cmd` dict from NLU and dispatches
    to the right agent.
    """

    def __init__(self):
        self.reminder = ReminderAgent()
        self.web = WebAgent()
        self.launcher = LauncherAgent()
        self.booking = BookingAgent()
        self.file = FileManagerAgent()
        self.process = ProcessManagerAgent()
        self.sleep = SleepAgent()
        self.browser = BrowserControlAgent()
        self.close = AppCloseAgent()
        self.mail = MailAgent()

    def handle(self, cmd: Dict[str, Any]) -> None:
        text = (cmd.get("original_text") or "")

        # File dialog continuation has highest priority
        if self.file.in_dialog():
            self.file.manage(cmd)
            return

        # If MailAgent is waiting for yes/no about reading body
        if getattr(self.mail, "pending_read", None):
            self.mail.handle(cmd)
            return

        # If BookingAgent is in conversational booking flow
        if self.booking.in_booking():
            self.booking.handle(cmd)
            return

        intent = (cmd.get("intent") or {}).get("label", "").lower()
        prefix = intent.split(".")[0] if "." in intent else intent
        agent_name = INTENT_AGENT_MAP.get(prefix)

        if not agent_name:
            print(f"No agent mapped for intent '{intent}'")
            return

        if agent_name == "reminder":
            self.reminder.create(cmd)
            return
        if agent_name == "web":
            self.web.search(cmd)
            return
        if agent_name == "launcher":
            self.launcher.launch(cmd)
            return
        if agent_name == "file":
            self.file.manage(cmd)
            return
        if agent_name == "process":
            self.process.handle(cmd)
            return
        if agent_name == "sleep":
            self.sleep.handle(cmd)
            return
        if agent_name == "browser":
            self.browser.control(cmd)
            return
        if agent_name == "booking":
            self.booking.handle(cmd)
            return
        if agent_name == "mail":
            self.mail.handle(cmd)
            return
        if agent_name == "close":
            self.close.handle(cmd)
            return

