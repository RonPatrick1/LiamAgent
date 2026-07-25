#!/usr/bin/env python3
"""HTTP entry point for the Patrick Messenger integration — see
agent/server.py for what this actually does and why.
"""

import os


def _load_dotenv(path=None):
    """Resolved against this script's own directory, not the process's
    cwd — same convention as LiamAgent.py/LiamGUI.py."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

from agent.server import main  # noqa: E402

if __name__ == "__main__":
    main()
