# agents/__init__.py

from .planner import Planner
from .reminder_agent import ReminderAgent
from .web_agent import WebAgent
from .launcher_agent import LauncherAgent
from .app_close_agent import AppCloseAgent
from .file_manager_agent import FileManagerAgent
from .process_manager_agent import ProcessManagerAgent
from .sleep_agent import SleepAgent
from .mail_agent import MailAgent
from .booking_agent import BookingAgent
from .browser_control_agent import BrowserControlAgent

__all__ = [
    "Planner",
    "ReminderAgent",
    "WebAgent",
    "LauncherAgent",
    "AppCloseAgent",
    "FileManagerAgent",
    "ProcessManagerAgent",
    "SleepAgent",
    "MailAgent",
    "BookingAgent",
    "BrowserControlAgent",
]

