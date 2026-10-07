"""A localhost-only TensorBoard subprocess; never terminates unrelated services."""

import os
import json
import socket
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse
import webbrowser
from pathlib import Path

from lung.engine.checkpoint import atomic_json


def healthy(url):
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return response.status == 200
    except (OSError, ValueError):
        return False


def launch_dashboard(log_root, status_path, open_browser=True):
    log_root = Path(log_root).resolve()
    registry_path = log_root / "dashboard_service.json"
    for registered in (Path(status_path), registry_path):
        if not registered.exists():
            continue
        try:
            record = json.loads(registered.read_text(encoding="utf-8"))
            url = record["url"]
            address = urlparse(url)
            if address.hostname != "127.0.0.1" or address.scheme != "http" or address.port not in range(6006, 6017):
                continue
            with urllib.request.urlopen(url + "/data/environment", timeout=2) as response:
                environment = json.load(response)
            actual_path = environment.get("data_location", "")
            if Path(actual_path).resolve() == log_root and healthy(url):
                record = {**record, "status": "running", "reused": True}
                atomic_json(status_path, record)
                if open_browser:
                    webbrowser.open(url + "/#scalars")
                return None, record
        except (OSError, ValueError, KeyError, TypeError):
            continue
    process = None
    for port in range(6006, 6017):
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        except OSError:
            continue
        output = Path(status_path).parent / "tensorboard.log"
        with output.open("ab") as stream:
            process = subprocess.Popen(
                [sys.executable, "-m", "tensorboard.main", "--logdir", str(log_root),
                 "--host", "127.0.0.1", "--port", str(port), "--reload_interval", "5"],
                stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                cwd=log_root.parent)
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 20
        while process.poll() is None and time.monotonic() < deadline:
            if healthy(url):
                record = {"status": "running", "pid": process.pid, "url": url,
                          "logdir": str(log_root), "reload_seconds": 5,
                          "browser_auto_reload": "Enable Reload data in TensorBoard settings"}
                atomic_json(status_path, record)
                atomic_json(registry_path, record)
                if open_browser:
                    webbrowser.open(url + "/#scalars")
                return process, record
            time.sleep(.2)
        # Only our just-created process can be stopped here.
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        break
    record = {"status": "unavailable", "reason": "No healthy TensorBoard service; inspect tensorboard.log"}
    atomic_json(status_path, record)
    return None, record
