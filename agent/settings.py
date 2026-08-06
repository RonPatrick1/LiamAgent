"""Local machine preferences for Liam — model choice, auto-confirm,
custom instructions. A plain JSON file next to the code, not the shared
MySQL DB: this is a per-machine app preference, not conversation data,
same reasoning as .liam_window_state.json.
"""

import json
import os

SETTINGS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".liam_settings.json",
)

# model=None means "use agent.llm.DEFAULT_MODEL" — kept as None rather than
# duplicating that constant here, so upgrading the default model upstream
# doesn't require touching a settings file too.
DEFAULTS = {
    "model": None,
    "auto_confirm": True,
    "custom_instructions": "",
    "show_external_sessions": True,
    "dialog_geometry": {},
    # Paned divider positions and panel visibility, restored on launch so
    # the window looks the way it did when it was last closed. None means
    # "never dragged/toggled yet" — the widget's own hardcoded construction
    # default applies instead of a guessed number here.
    "sidebar_paned_position": None,
    "content_paned_position": None,
    "right_stack_position": None,
    "artifacts_split_position": None,
    "artifacts_visible": False,
    "plan_panel_visible": True,
}


def load():
    settings = dict(DEFAULTS)
    try:
        with open(SETTINGS_FILE) as f:
            saved = json.load(f)
        for key in DEFAULTS:
            if key in saved:
                settings[key] = saved[key]
    except Exception:
        pass
    return settings


def save(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump({key: settings.get(key, DEFAULTS[key]) for key in DEFAULTS}, f)
    except Exception as exc:
        print(f"[settings] failed to save: {exc}")
