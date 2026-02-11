# agents/process_manager_agent.py

import psutil

from utils import open_with, speak


class ProcessManagerAgent:
    def open_task_manager(self):
        if open_with(
            [
                "gnome-system-monitor",
                "mate-system-monitor",
                "ksysguard",
                "plasma-systemmonitor",
                "xfce4-taskmanager",
            ]
        ):
            print("(process) Opened Task Manager.")
            speak("Opening system monitor.")
        else:
            print("(process) Could not open Task Manager. Install gnome-system-monitor.")
            speak("Could not open task manager.")

    def analyze_system_load(self):
        cpu_usage = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory()
        procs = []
        for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                procs.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        procs = sorted(
            procs,
            key=lambda x: (x.get("cpu_percent", 0) + x.get("memory_percent", 0)),
            reverse=True,
        )[:5]
        if cpu_usage > 80 or mem.percent > 85:
            culprit = procs[0] if procs else None
            msg = f"High load — CPU {cpu_usage}%, Memory {mem.percent}%."
            if culprit:
                msg += (
                    f" Top: {culprit.get('name')} (PID {culprit.get('pid')}) "
                    f"CPU {culprit.get('cpu_percent')}%."
                )
        else:
            msg = f"System OK — CPU {cpu_usage}%, Memory {mem.percent}%."
        print("(process)", msg)
        speak(msg)
        if procs:
            for p in procs:
                print(
                    f"  {p.get('pid'):<6} {p.get('name'):<18} "
                    f"CPU {p.get('cpu_percent', 0):>5.1f}% "
                    f"MEM {p.get('memory_percent', 0):>5.1f}%"
                )

    def top_cpu(self, n=10):
        table = "PID   NAME               %CPU  %MEM\n"
        out = []
        for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                out.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        out = sorted(out, key=lambda x: x.get("cpu_percent", 0), reverse=True)[:n]
        print("\n(process) Top CPU processes:")
        print(table)
        for p in out:
            print(
                f"{p.get('pid'):>5} {p.get('name')[:18]:<18} "
                f"{p.get('cpu_percent', 0):>5.1f} {p.get('memory_percent', 0):>5.1f}"
            )

    def top_mem(self, n=10):
        table = "PID   NAME               %CPU  %MEM\n"
        out = []
        for p in psutil.process_iter(attrs=["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                out.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        out = sorted(out, key=lambda x: x.get("memory_percent", 0), reverse=True)[:n]
        print("\n(process) Top Memory processes:")
        print(table)
        for p in out:
            print(
                f"{p.get('pid'):>5} {p.get('name')[:18]:<18} "
                f"{p.get('cpu_percent', 0):>5.1f} {p.get('memory_percent', 0):>5.1f}"
            )

    def handle(self, cmd):
        t = (cmd.get("original_text") or "").lower()
        if any(p in t for p in ["task manager", "system monitor", "open task manager", "show task manager"]):
            self.open_task_manager()
            return
        if any(p in t for p in ["top cpu", "cpu processes", "high cpu"]):
            self.top_cpu()
            return
        if any(p in t for p in ["top memory", "top ram", "memory processes", "high memory"]):
            self.top_mem()
            return
        if any(p in t for p in ["which process makes it slow", "slow processes", "what makes it slow", "why is it slow", "lag"]):
            self.analyze_system_load()
            return
        self.top_cpu()

