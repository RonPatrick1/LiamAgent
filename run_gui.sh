#!/bin/bash
# Launcher wrapper — LiamGUI.py loads .env via a relative path, so this
# needs to cd into the project directory first regardless of where it's
# launched from (app grid launches don't share a shell's cwd).
cd /var/www/LiamAgent
exec python3 LiamGUI.py "$@"
