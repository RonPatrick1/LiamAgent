#!/usr/bin/env python3
"""Native GTK3 desktop UI for the Liam agent.

GTK3, not GTK4: GTK4 dropped Gtk.Window.move()/resize()/get_position() from
its public API entirely (Wayland forbids apps from doing this), which is
why restoring window position/size on close/reopen turned into a losing
fight with xdotool and then raw Xlib. GTK3 still has those calls natively,
plus real GdkWindowState bits (MAXIMIZED, LEFT_TILED, RIGHT_TILED,
TOP_TILED, BOTTOM_TILED) reported directly by the window manager — the same
mechanism FredPlayerForAliens and Patrick Messenger's own Linux window-
placement code (linux/runner/my_application.cc) already use. This mirrors
that approach directly instead of working around GTK4's removed APIs.
"""

import base64
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime

import requests

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("GdkPixbuf", "2.0")
try:
    gi.require_version("GdkX11", "3.0")
    from gi.repository import GdkX11
except (ValueError, ImportError):
    GdkX11 = None
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib, Pango  # noqa: E402

try:
    from Xlib.display import Display
except ImportError:
    Display = None


def _load_dotenv(path=None):
    """Resolved against this script's own directory, not the process's
    cwd — Liam can be launched from any folder (each becomes its own
    session), and .env always lives alongside the code."""
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

from agent import memory, routines, ssh_secrets  # noqa: E402
from agent import tools as agent_tools  # noqa: E402
from agent import settings as liam_settings  # noqa: E402
from agent.core import Agent  # noqa: E402
from agent.llm import DEFAULT_MODEL  # noqa: E402

APP_ID = "com.ronpatrick.Liam"

# How many past messages to replay into the chat view when switching to a
# thread — independent of agent.core.HISTORY_LIMIT, which controls how much
# history actually gets fed back into the model's context.
REPLAY_LIMIT = 50

WINDOW_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".liam_window_state.json")

# Pasted images are saved here rather than base64-dumped into the MySQL
# messages table — that column is meant for text, and a handful of pasted
# screenshots would bloat it fast. The model still sees the actual image
# data for the turn it's pasted in (via Agent.step's images= param); only
# persisted history gets a plain-text placeholder instead of the bytes.
PASTE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".liam_pastes")

# Images Liam references via Markdown syntax (![alt](url), from
# image_search results) get downloaded here once and cached by URL hash,
# so re-showing the same image across turns/threads doesn't re-fetch it.
DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".liam_downloads")

# Matches agent/core.py's system-prompt instruction to use standard
# Markdown image syntax for anything found via image_search.
IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")

EXTERNAL_SESSION_PREFIXES = ("/matrix-room/", "/fredplayer-device/")

SAVE_DELAY_MS = 350
RESTORE_SETTLE_MS = 500

_TILE_STATE_BITS = (
    Gdk.WindowState.TILED | Gdk.WindowState.LEFT_TILED | Gdk.WindowState.RIGHT_TILED
    | Gdk.WindowState.TOP_TILED | Gdk.WindowState.BOTTOM_TILED
)

_xlib_display = None


def _get_xlib_display():
    global _xlib_display
    if _xlib_display is None and Display is not None:
        try:
            _xlib_display = Display()
        except Exception:
            _xlib_display = False
    return _xlib_display or None


def _monitor_index_at_point(display, x, y):
    """Which monitor (by index) currently contains this point, or -1 if
    none does (e.g. the monitor layout changed since this was saved)."""
    for i in range(display.get_n_monitors()):
        geo = display.get_monitor(i).get_geometry()
        if geo.x <= x < geo.x + geo.width and geo.y <= y < geo.y + geo.height:
            return i
    return -1


def _is_external_session(session):
    """Whether a session belongs to one of Liam's external chat bridges.

    Use the server-owned folder namespace rather than the display title so
    an ordinary desktop thread named "Matrix" or "FredPlayer" is never
    hidden accidentally.
    """
    folder_path = session.get("folder_path", "")
    return any(folder_path.startswith(prefix) for prefix in EXTERNAL_SESSION_PREFIXES)


def _routine_display_prompt(prompt):
    """Hide the execution wrapper while retaining the user's real task."""
    prompt = prompt or ""
    original_marker = "Original request:\n"
    if original_marker in prompt:
        return prompt.split(original_marker, 1)[1].strip()
    exact_marker = "Return exactly this message:"
    if exact_marker in prompt:
        return prompt.split(exact_marker, 1)[1].strip()
    return prompt.strip()


def _finished_once_status(routine, now=None):
    """Return Completed/Expired for one-time routines that cannot resume."""
    if routine.get("schedule_kind") != "once":
        return None
    if routine.get("last_run_at") is not None:
        return "Completed"
    try:
        run_at = datetime.strptime(routine["schedule_value"], "%Y-%m-%d %H:%M:%S")
    except (KeyError, TypeError, ValueError):
        return "Invalid schedule"
    return "Expired" if run_at <= (now or datetime.now()) else None


def _move_window(window, x, y):
    """Match Patrick Messenger's scaled XMoveWindow path on X11.

    gtk_window_move() is retained as the pre-realization and non-X11
    fallback.  Once mapped on X11, using the real XID avoids GTK3's
    inconsistent coordinate scaling on HiDPI monitor layouts.
    """
    gdk_window = window.get_window()
    xdisplay = _get_xlib_display()
    if gdk_window is not None and GdkX11 is not None and xdisplay is not None:
        try:
            xid = GdkX11.X11Window.get_xid(gdk_window)
            scale = gdk_window.get_scale_factor() or 1
            xwindow = xdisplay.create_resource_object("window", xid)
            xwindow.configure(x=int(x * scale), y=int(y * scale))
            xdisplay.flush()
            return
        except (TypeError, AttributeError):
            # A non-X11 GdkWindow reaches the portable fallback below.
            pass
        except Exception:
            pass
    window.move(int(x), int(y))


def _default_window_placement():
    return {
        "valid": False,
        "x": 0,
        "y": 0,
        "width": 960,
        "height": 860,
        "monitor": -1,
        "tile_side": 0,
        "maximized": False,
        "tiled": False,
    }


def _load_window_placement():
    placement = _default_window_placement()
    try:
        with open(WINDOW_STATE_FILE) as f:
            saved = json.load(f)
    except Exception:
        return placement

    for key in placement:
        if key in saved:
            placement[key] = saved[key]

    # Migrate the earlier GTK4/first GTK3 state format.
    if "valid" not in saved:
        placement["valid"] = saved.get("x") is not None and saved.get("y") is not None
    if "tile_side" not in saved:
        placement["tile_side"] = {"left": -1, "right": 1}.get(saved.get("tile_edge"), 0)

    if placement["width"] < 480 or placement["height"] < 360:
        placement["valid"] = False
    return placement


class LiamWindow(Gtk.ApplicationWindow):
    def __init__(self, app, model, auto_confirm):
        super().__init__(application=app, title="Liam", default_width=960, default_height=860)
        self.set_icon_name(APP_ID)
        self.model = model
        self.auto_confirm = auto_confirm
        self.settings = liam_settings.load()
        self.agent = None
        self.session_id = None
        self.sessions = []
        self.busy = False
        self._thinking_active = False
        self._thinking_timer_id = None
        self._thinking_anchor_mark = None
        self._thinking_text_mark = None
        self._thinking_started_at = None
        self._thinking_label = None

        # Patrick Messenger's placement lifecycle, kept deliberately
        # separate from the rest of the UI state.
        self._placement = _load_window_placement()
        self._save_timeout_id = 0
        self._restore_timeout_id = 0
        self._restoring_placement = self._placement["valid"]
        self._post_map_restore_done = False

        headerbar = Gtk.HeaderBar()
        headerbar.set_show_close_button(True)
        headerbar.set_title("Liam")
        self.headerbar = headerbar
        self.set_titlebar(headerbar)

        new_button = Gtk.Button.new_from_icon_name("list-add-symbolic", Gtk.IconSize.BUTTON)
        new_button.set_tooltip_text("New thread")
        new_button.connect("clicked", self._on_new_thread)
        headerbar.pack_end(new_button)

        settings_button = Gtk.Button.new_from_icon_name("emblem-system-symbolic", Gtk.IconSize.BUTTON)
        settings_button.set_tooltip_text("Customize")
        settings_button.connect("clicked", self._open_settings_dialog)
        headerbar.pack_end(settings_button)

        routines_button = Gtk.Button.new_from_icon_name("alarm-symbolic", Gtk.IconSize.BUTTON)
        routines_button.set_tooltip_text("Routines")
        routines_button.connect("clicked", self._open_routines_dialog)
        headerbar.pack_end(routines_button)

        lessons_button = Gtk.Button.new_from_icon_name("view-list-bullet-symbolic", Gtk.IconSize.BUTTON)
        lessons_button.set_tooltip_text("Lessons")
        lessons_button.connect("clicked", self._open_lessons_dialog)
        headerbar.pack_end(lessons_button)

        self.artifacts_toggle = Gtk.ToggleButton()
        self.artifacts_toggle.get_style_context().add_class("liam-toggle")
        self.artifacts_toggle.set_image(
            Gtk.Image.new_from_icon_name("view-paged-symbolic", Gtk.IconSize.BUTTON)
        )
        self.artifacts_toggle.set_tooltip_text("Artifacts")
        self.artifacts_toggle.connect("toggled", self._on_artifacts_toggled)
        headerbar.pack_end(self.artifacts_toggle)

        self.spinner = Gtk.Spinner()
        self.spinner.set_visible(False)
        headerbar.pack_end(self.spinner)

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_position(220)
        paned.set_wide_handle(True)
        self.add(paned)

        # --- sidebar: one thread per folder, like Recents ---
        self.session_list = Gtk.ListBox()
        self.session_list.get_style_context().add_class("liam-select-list")
        self._install_liam_css()
        self.session_list.connect("row-activated", self._on_row_activated)
        self.session_list.connect("button-press-event", self._on_session_list_button_press)
        session_scroller = Gtk.ScrolledWindow()
        session_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        session_scroller.set_vexpand(True)
        session_scroller.add(self.session_list)

        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.external_sessions_toggle = Gtk.CheckButton(label="Show Matrix / FredPlayer")
        self.external_sessions_toggle.get_style_context().add_class("liam-toggle")
        self.external_sessions_toggle.set_tooltip_text(
            "Show conversations created by Matrix and FredPlayer"
        )
        self.external_sessions_toggle.set_active(self.settings["show_external_sessions"])
        self.external_sessions_toggle.set_margin_start(10)
        self.external_sessions_toggle.set_margin_end(10)
        self.external_sessions_toggle.set_margin_top(6)
        self.external_sessions_toggle.set_margin_bottom(4)
        self.external_sessions_toggle.connect("toggled", self._on_external_sessions_toggled)
        sidebar_box.pack_start(self.external_sessions_toggle, False, False, 0)
        sidebar_box.pack_start(session_scroller, True, True, 0)
        paned.pack1(sidebar_box, resize=False, shrink=False)

        # --- content: the chat view ---
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        # A single continuous text buffer for the whole conversation —
        # not one widget per message — so drag-selection and copy work
        # across multiple messages at once, the way normal text does.
        # Separate Gtk.Label "bubbles" can only ever select within
        # themselves; there's no way to span widgets with a text drag.
        self.buffer = Gtk.TextBuffer()
        self.textview = Gtk.TextView(buffer=self.buffer)
        self.textview.get_style_context().add_class("liam-chat-view")
        self.textview.set_editable(False)
        self.textview.set_cursor_visible(False)
        self.textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.textview.set_left_margin(12)
        self.textview.set_right_margin(12)
        self.textview.set_top_margin(12)
        self.textview.set_bottom_margin(12)

        self.tag_user = self.buffer.create_tag(
            "user", foreground="#3584e4", weight=Pango.Weight.BOLD,
            justification=Gtk.Justification.RIGHT,
        )
        self.tag_assistant = self.buffer.create_tag("assistant", justification=Gtk.Justification.LEFT)
        self.tag_status = self.buffer.create_tag(
            "status", foreground="#888888", family="monospace",
            style=Pango.Style.ITALIC, justification=Gtk.Justification.LEFT,
        )
        self.tag_bold = self.buffer.create_tag("bold", weight=Pango.Weight.BOLD)
        self.tag_code = self.buffer.create_tag("code", family="monospace")

        self.scroller = Gtk.ScrolledWindow()
        self.scroller.set_vexpand(True)
        self.scroller.add(self.textview)
        content_box.pack_start(self.scroller, True, True, 0)

        bottom = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        bottom.set_margin_top(8)
        bottom.set_margin_bottom(12)
        bottom.set_margin_start(12)
        bottom.set_margin_end(12)

        # A pasted image (Ctrl+V into the entry with an image on the
        # clipboard) sits here as a small thumbnail + remove button until
        # sent, rather than vanishing silently or being pasted as garbage
        # text — Gtk.Entry has no image-paste support of its own.
        self._pending_image_path = None
        self.pending_image_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.pending_image_box.set_visible(False)
        self.pending_image_view = Gtk.Image()
        self.pending_image_box.pack_start(self.pending_image_view, False, False, 0)
        remove_pending_button = Gtk.Button.new_from_icon_name("window-close-symbolic", Gtk.IconSize.BUTTON)
        remove_pending_button.set_tooltip_text("Remove pasted image")
        remove_pending_button.connect("clicked", lambda _b: self._clear_pending_image())
        self.pending_image_box.pack_start(remove_pending_button, False, False, 0)
        bottom.pack_start(self.pending_image_box, False, False, 0)

        # A Gtk.Entry can't hold more than one line at all — Ctrl+Enter for
        # a newline needs a real multi-line widget. GTK delivers full key
        # events with modifier state, unlike a terminal, so Enter-vs-
        # Ctrl+Enter is fully reliable here.
        self.entry = Gtk.TextView()
        self.entry.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.entry.connect("key-press-event", self._on_entry_key_press)
        self.entry.connect("paste-clipboard", self._on_entry_paste_clipboard)
        entry_scroller = Gtk.ScrolledWindow()
        entry_scroller.set_hexpand(True)
        entry_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        entry_scroller.set_min_content_height(36)
        entry_scroller.set_max_content_height(100)
        entry_scroller.add(self.entry)
        bottom.pack_start(entry_scroller, True, True, 0)

        self.send_button = Gtk.Button(label="Send")
        self.send_button.get_style_context().add_class("liam-accent-button")
        self.send_button.connect("clicked", self._on_send)
        bottom.pack_start(self.send_button, False, False, 0)

        content_box.pack_start(bottom, False, False, 0)

        # --- artifacts: generated files/code for the current thread ---
        self._current_artifact = None
        self.artifacts_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.artifacts_box.set_size_request(260, -1)
        self.artifacts_box.set_visible(False)

        artifacts_label = Gtk.Label(label="Artifacts", xalign=0)
        artifacts_label.set_margin_top(8)
        artifacts_label.set_margin_bottom(4)
        artifacts_label.set_margin_start(10)
        artifacts_label.get_style_context().add_class("dim-label")
        self.artifacts_box.pack_start(artifacts_label, False, False, 0)

        self.artifacts_list = Gtk.ListBox()
        self.artifacts_list.get_style_context().add_class("liam-select-list")
        self.artifacts_list.connect("row-selected", self._on_artifact_selected)
        artifacts_list_scroller = Gtk.ScrolledWindow()
        artifacts_list_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        artifacts_list_scroller.set_size_request(-1, 160)
        artifacts_list_scroller.add(self.artifacts_list)
        self.artifacts_box.pack_start(artifacts_list_scroller, False, False, 0)

        self.artifact_buffer = Gtk.TextBuffer()
        artifact_view = Gtk.TextView(buffer=self.artifact_buffer)
        artifact_view.set_editable(False)
        artifact_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        artifact_view.set_left_margin(8)
        artifact_view.set_right_margin(8)
        artifact_scroller = Gtk.ScrolledWindow()
        artifact_scroller.set_vexpand(True)
        artifact_scroller.add(artifact_view)
        self.artifacts_box.pack_start(artifact_scroller, True, True, 0)

        artifact_actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        artifact_actions.set_margin_top(6)
        artifact_actions.set_margin_bottom(8)
        artifact_actions.set_margin_start(8)
        artifact_actions.set_margin_end(8)
        self.artifact_save_button = Gtk.Button(label="Save As…")
        self.artifact_save_button.connect("clicked", self._on_artifact_save_as)
        self.artifact_save_button.set_sensitive(False)
        artifact_actions.pack_start(self.artifact_save_button, True, True, 0)
        self.artifacts_box.pack_start(artifact_actions, False, False, 0)

        inner_paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        inner_paned.pack1(content_box, resize=True, shrink=False)
        inner_paned.pack2(self.artifacts_box, resize=False, shrink=True)
        paned.pack2(inner_paned, resize=True, shrink=False)

        self.connect("configure-event", self._on_configure_event)
        self.connect("window-state-event", self._on_window_state_event)
        self.connect("delete-event", self._on_delete_event)
        self.connect("map-event", self._on_map_event)

        self._bootstrap_sessions()

    # --- window position/size/monitor memory ---
    #
    # Mirrors Patrick Messenger directly: store the settled outer-frame
    # position plus GTK's logical window size, replay those exact values,
    # and prevent restore-generated configure events from corrupting the
    # saved placement while Mutter settles the window.

    def _placement_monitor(self, display):
        placement = self._placement
        center_x = placement["x"] + placement["width"] // 2
        center_y = placement["y"] + placement["height"] // 2
        index = _monitor_index_at_point(display, center_x, center_y)
        if index < 0:
            saved_index = placement["monitor"]
            if isinstance(saved_index, int) and 0 <= saved_index < display.get_n_monitors():
                index = saved_index
        if index >= 0:
            return display.get_monitor(index)
        primary = display.get_primary_monitor()
        if primary is not None:
            return primary
        return display.get_monitor(0) if display.get_n_monitors() > 0 else None

    def _write_window_placement(self):
        if not self._placement["valid"]:
            return
        try:
            with open(WINDOW_STATE_FILE, "w") as f:
                json.dump(self._placement, f)
        except Exception:
            pass

    def _save_window_timeout(self):
        self._save_timeout_id = 0
        self._write_window_placement()
        return False

    def _schedule_window_state_save(self):
        if self._restoring_placement:
            return
        if self._save_timeout_id:
            GLib.source_remove(self._save_timeout_id)
        self._save_timeout_id = GLib.timeout_add(SAVE_DELAY_MS, self._save_window_timeout)

    def _on_configure_event(self, _widget, _event):
        gdk_window = self.get_window()
        if gdk_window is None:
            return False
        wstate = gdk_window.get_state()
        if self._restoring_placement:
            return False
        if not (wstate & (Gdk.WindowState.MAXIMIZED | Gdk.WindowState.FULLSCREEN)):
            frame = gdk_window.get_frame_extents()
            width, height = self.get_size()
            placement = self._placement
            placement.update({
                "valid": True,
                "x": frame.x,
                "y": frame.y,
                "width": width,
                "height": height,
            })
            display = self.get_display()
            placement["monitor"] = _monitor_index_at_point(
                display, frame.x + width // 2, frame.y + height // 2,
            )
            if placement["tiled"] and placement["monitor"] >= 0:
                geometry = display.get_monitor(placement["monitor"]).get_geometry()
                center_x = frame.x + width // 2
                placement["tile_side"] = -1 if center_x < geometry.x + geometry.width // 2 else 1
        self._schedule_window_state_save()
        return False

    def _on_window_state_event(self, _widget, event):
        placement = self._placement
        placement["maximized"] = bool(event.new_window_state & Gdk.WindowState.MAXIMIZED)
        placement["tiled"] = (
            not placement["maximized"]
            and bool(event.new_window_state & _TILE_STATE_BITS)
        )
        if placement["tiled"] and not self._restoring_placement:
            display = self.get_display()
            center_x = placement["x"] + placement["width"] // 2
            center_y = placement["y"] + placement["height"] // 2
            placement["monitor"] = _monitor_index_at_point(display, center_x, center_y)
            if placement["monitor"] >= 0:
                geometry = display.get_monitor(placement["monitor"]).get_geometry()
                placement["tile_side"] = -1 if center_x < geometry.x + geometry.width // 2 else 1
        self._schedule_window_state_save()
        return False

    def _on_delete_event(self, _widget, _event):
        if self._save_timeout_id:
            GLib.source_remove(self._save_timeout_id)
            self._save_timeout_id = 0
        self._write_window_placement()
        return False  # allow the close to proceed

    def _finish_window_restore(self):
        self._restore_timeout_id = 0
        self._restoring_placement = False
        return False

    def _on_map_event(self, _widget, _event):
        if self._placement["valid"] and not self._post_map_restore_done:
            self._post_map_restore_done = True
            # GNOME may ignore the pre-map position. Patrick repeats the
            # restore after Flutter's first frame; map-event is the native
            # GTK equivalent for this non-Flutter window.
            self.restore_saved_placement()
            self._restore_timeout_id = GLib.timeout_add(
                RESTORE_SETTLE_MS, self._finish_window_restore,
            )
        return False

    def restore_saved_placement(self):
        placement = self._placement
        if not placement["valid"]:
            self.set_default_size(960, 860)
            return

        display = self.get_display()
        monitor = self._placement_monitor(display)
        if monitor is None:
            self.set_default_size(placement["width"], placement["height"])
            _move_window(self, placement["x"], placement["y"])
            return

        workarea = monitor.get_workarea()
        width, height = placement["width"], placement["height"]
        x, y = placement["x"], placement["y"]
        saved_monitor = _monitor_index_at_point(display, x + width // 2, y + height // 2)
        if not placement["tiled"] or saved_monitor < 0:
            width = min(width, workarea.width)
            height = min(height, workarea.height)
            x = max(workarea.x, min(x, workarea.x + workarea.width - width))
            y = max(workarea.y, min(y, workarea.y + workarea.height - height))

        self.set_default_size(width, height)
        self.resize(width, height)
        _move_window(self, x, y)
        if placement["maximized"]:
            self.maximize()

    def flush_window_placement(self):
        if self._save_timeout_id:
            GLib.source_remove(self._save_timeout_id)
            self._save_timeout_id = 0
        if self._restore_timeout_id:
            GLib.source_remove(self._restore_timeout_id)
            self._restore_timeout_id = 0
        self._write_window_placement()

    # --- session/thread management ---

    def _install_liam_css(self):
        """Override the system theme's default accent color (Ubuntu's Yaru
        defaults it to orange/green — including text selection inside a
        plain GtkTextView) in favor of colors actually sampled from
        icons/liam.png — its glowing outline averages to #32a2e2, its
        dark metal head to #1c2c4a. Scoped to dedicated classes
        (.liam-select-list, .liam-toggle, .liam-accent-button,
        .liam-chat-view) rather
        than overriding GTK's own "suggested-action"/selection styling
        globally, which would leak into every other GTK3 app's widgets in
        this session (add_provider_for_screen is X-session-wide, not
        limited to this process), not just Liam's."""
        provider = Gtk.CssProvider()
        provider.load_from_data(b"""
            list.liam-select-list row:selected {
                background-color: rgba(70, 110, 150, 0.35);
                background-image: none;
                color: inherit;
            }
            switch.liam-toggle:checked {
                background-color: #2f6f9e;
                background-image: none;
                border-color: #1c2c4a;
            }
            checkbutton.liam-toggle check:checked {
                background-color: #2f6f9e;
                background-image: none;
                border-color: #1c2c4a;
                color: #ffffff;
            }
            button.liam-toggle:checked {
                background-color: #2f6f9e;
                background-image: none;
                border-color: #1c2c4a;
                color: #ffffff;
            }
            textview.liam-chat-view,
            textview.liam-chat-view text {
                background-color: #000000;
                background-image: none;
                color: #e6e6e6;
            }
            textview.liam-chat-view text selection {
                background-color: #2f6f9e;
                color: #ffffff;
            }
            button.liam-accent-button {
                background-color: #2f6f9e;
                background-image: none;
                color: #ffffff;
                border-color: #1c2c4a;
            }
            button.liam-accent-button:hover {
                background-color: #3a82b5;
                background-image: none;
            }
            button.liam-accent-button:active {
                background-color: #255a80;
                background-image: none;
            }
        """)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _refresh_session_list(self):
        sessions = memory.list_sessions()
        if not self.settings["show_external_sessions"]:
            sessions = [session for session in sessions if not _is_external_session(session)]
        self.sessions = sessions
        for child in list(self.session_list.get_children()):
            self.session_list.remove(child)

        # Sessions already arrive pinned-first, then grouped (named groups
        # before ungrouped) — insert a plain header row whenever the group
        # changes, matching a "Move to group" sidebar without needing a
        # heavier custom header-func setup.
        last_group_id = "__unset__"
        for session in self.sessions:
            group_id = session.get("group_id")
            if group_id != last_group_id and not session.get("pinned"):
                header_row = Gtk.ListBoxRow()
                header_row.set_selectable(False)
                header_row.set_activatable(False)
                header_label = Gtk.Label(
                    label=(session.get("group_name") or "Ungrouped").upper(), xalign=0,
                )
                header_label.set_ellipsize(Pango.EllipsizeMode.END)
                header_label.set_margin_top(8)
                header_label.set_margin_bottom(2)
                header_label.set_margin_start(10)
                header_label.get_style_context().add_class("dim-label")
                header_row.add(header_label)
                self.session_list.add(header_row)
                last_group_id = group_id

            row_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            row_box.set_margin_top(6)
            row_box.set_margin_bottom(6)
            row_box.set_margin_start(10)
            row_box.set_margin_end(10)
            title_text = GLib.markup_escape_text(session["title"])
            if session.get("pinned"):
                title_text = f"📌 {title_text}"
            if session.get("unread"):
                title_text = f"<b>{title_text}</b>"
            title_label = Gtk.Label(xalign=0)
            title_label.set_markup(title_text)
            # Without ellipsizing, a Gtk.Label's minimum width equals its
            # full unwrapped text width — the sidebar (and the Paned
            # divider controlling it) can never be dragged narrower than
            # whatever the single longest title/path in the list happens
            # to be. Ellipsizing gives the label a small real minimum
            # width instead, so the divider is free to go as narrow as
            # you actually want.
            title_label.set_ellipsize(Pango.EllipsizeMode.END)
            extra_count = len(memory.list_session_folders(session["id"]))
            subtitle_text = session["folder_path"]
            if extra_count:
                subtitle_text += f"  (+{extra_count} folder{'s' if extra_count != 1 else ''})"
            subtitle_label = Gtk.Label(label=subtitle_text, xalign=0)
            subtitle_label.set_ellipsize(Pango.EllipsizeMode.END)
            subtitle_label.get_style_context().add_class("dim-label")
            row_box.pack_start(title_label, False, False, 0)
            row_box.pack_start(subtitle_label, False, False, 0)
            row = Gtk.ListBoxRow()
            row.add(row_box)
            row.session_data = session
            self.session_list.add(row)
        self.session_list.show_all()

    def _on_external_sessions_toggled(self, toggle):
        show_external = toggle.get_active()
        self.settings["show_external_sessions"] = show_external
        liam_settings.save(self.settings)

        current_session = memory.get_session(self.session_id) if self.session_id else None
        current_will_be_hidden = (
            not show_external
            and current_session is not None
            and _is_external_session(current_session)
        )

        self._refresh_session_list()
        if not current_will_be_hidden:
            if self.session_id is not None:
                self._select_row_for_session(self.session_id)
            return

        # Keep an actual thread selected when the filter hides the one that
        # was on screen. If every normal thread was archived, revive Liam's
        # own working-directory thread rather than leaving an empty sidebar.
        if not self.sessions:
            fallback_id = memory.get_or_create_session(os.getcwd())
            memory.set_archived(fallback_id, False)
            self._refresh_session_list()
        if not self.sessions:
            return
        next_session = self.sessions[0]
        self._switch_to(next_session["folder_path"], next_session["title"])
        self._select_row_for_session(next_session["id"])

    def _select_row_for_session(self, session_id):
        for row in self.session_list.get_children():
            if getattr(row, "session_data", {}).get("id") == session_id:
                self.session_list.select_row(row)
                return

    def _bootstrap_sessions(self):
        self._refresh_session_list()
        if not self.sessions:
            fallback_id = memory.get_or_create_session(os.getcwd())
            memory.set_archived(fallback_id, False)
            self._refresh_session_list()
        session = self.sessions[0]
        agent, history = self._build_agent_and_history(session["id"], session["folder_path"])
        self._apply_session(session["id"], session["folder_path"], session["title"], agent, history)
        self._select_row_for_session(session["id"])

    def _build_agent_and_history(self, session_id, folder_path):
        extra_folders = [f["folder_path"] for f in memory.list_session_folders(session_id)]
        agent = Agent(
            model=self.model, auto_confirm=self.auto_confirm,
            workdir=folder_path, session_id=session_id, extra_folders=extra_folders,
            custom_instructions=self.settings["custom_instructions"],
            channel="gui", actor_id="local-owner", is_owner=True,
            learning_enabled=True,
        )
        history = memory.load_recent_messages(limit=REPLAY_LIMIT, session_id=session_id)
        return agent, history

    def _apply_session(self, session_id, folder_path, title, agent, history):
        self.session_id = session_id
        self.agent = agent
        agent.on_tool_call = self._on_tool_call
        agent.on_status = self._on_status
        agent.on_confirm = self._on_confirm

        memory.set_unread(session_id, False)

        self.buffer.set_text("")
        self.headerbar.set_title(title)

        replayed = False
        for msg in history:
            if msg["role"] not in ("user", "assistant"):
                continue
            self._append_message(msg["content"], msg["role"], use_markup=(msg["role"] == "assistant"))
            replayed = True
        if not replayed:
            self._append_message("Liam agent ready. Ask away.", "status")

        if self.artifacts_toggle.get_active():
            self._refresh_artifacts_list()

    def _switch_to(self, folder_path, title=None):
        """Entry point for both clicking an existing thread and creating a
        new one — get_or_create_session makes both the same operation.
        The actual work (DB calls, building a fresh Agent) runs in a
        background thread so the window stays responsive while it
        happens, instead of freezing on the GTK main thread."""
        if self.busy:
            return
        display_name = title or os.path.basename(os.path.abspath(os.path.expanduser(folder_path)).rstrip("/")) or folder_path
        self._set_busy(True)
        self.headerbar.set_title(f"Loading {display_name}…")
        self.spinner.set_visible(True)
        self.spinner.start()
        is_new = os.path.abspath(os.path.expanduser(folder_path)) not in {s["folder_path"] for s in self.sessions}
        threading.Thread(target=self._switch_worker, args=(folder_path, title, is_new), daemon=True).start()

    def _switch_worker(self, folder_path, title, is_new):
        session_id = memory.get_or_create_session(folder_path, title=title)
        agent, history = self._build_agent_and_history(session_id, folder_path)
        resolved_title = title or os.path.basename(os.path.abspath(os.path.expanduser(folder_path)).rstrip("/")) or folder_path
        GLib.idle_add(self._finish_switch, session_id, folder_path, resolved_title, agent, history, is_new)

    def _finish_switch(self, session_id, folder_path, title, agent, history, is_new):
        self._apply_session(session_id, folder_path, title, agent, history)
        if is_new:
            self._refresh_session_list()
            self._select_row_for_session(session_id)
        self.spinner.stop()
        self.spinner.set_visible(False)
        self._set_busy(False)
        return False

    def _on_row_activated(self, _listbox, row):
        session = getattr(row, "session_data", None)
        if session:
            self._switch_to(session["folder_path"], session["title"])

    def _on_session_list_button_press(self, listbox, event):
        if event.button != 3:  # right-click only
            return False
        row = listbox.get_row_at_y(int(event.y))
        if row is None or not getattr(row, "session_data", None):
            return False
        listbox.select_row(row)
        session = row.session_data
        menu = Gtk.Menu()

        open_in_item = Gtk.MenuItem(label="Open in")
        open_in_submenu = Gtk.Menu()
        new_window_item = Gtk.MenuItem(label="New window")
        new_window_item.connect("activate", lambda _item: self._open_in_new_window(session))
        open_in_submenu.append(new_window_item)
        file_manager_item = Gtk.MenuItem(label="File manager")
        file_manager_item.connect("activate", lambda _item: self._open_in_file_manager(session))
        open_in_submenu.append(file_manager_item)
        open_in_item.set_submenu(open_in_submenu)
        menu.append(open_in_item)
        menu.append(Gtk.SeparatorMenuItem())

        pin_item = Gtk.MenuItem(label="Unpin" if session.get("pinned") else "Pin")
        pin_item.connect("activate", lambda _item: self._toggle_pin(session))
        menu.append(pin_item)

        unread_item = Gtk.MenuItem(label="Mark as read" if session.get("unread") else "Mark as unread")
        unread_item.connect("activate", lambda _item: self._toggle_unread(session))
        menu.append(unread_item)

        rename_item = Gtk.MenuItem(label="Rename…")
        rename_item.connect("activate", lambda _item: self._rename_session(session))
        menu.append(rename_item)

        fork_item = Gtk.MenuItem(label="Fork")
        fork_item.connect("activate", lambda _item: self._fork_session(session))
        menu.append(fork_item)

        add_folder_item = Gtk.MenuItem(label="Add folder…")
        add_folder_item.connect("activate", lambda _item: self._add_session_folder(session))
        menu.append(add_folder_item)
        menu.append(Gtk.SeparatorMenuItem())

        move_to_group_item = Gtk.MenuItem(label="Move to group")
        move_to_group_item.set_submenu(self._build_group_submenu(session))
        menu.append(move_to_group_item)
        menu.append(Gtk.SeparatorMenuItem())

        archive_item = Gtk.MenuItem(label="Archive")
        archive_item.connect("activate", lambda _item: self._archive_session(session))
        menu.append(archive_item)

        delete_item = Gtk.MenuItem(label="Delete")
        delete_item.connect("activate", lambda _item: self._delete_session(session))
        menu.append(delete_item)

        menu.show_all()
        menu.popup_at_pointer(event)
        return True

    def _open_in_new_window(self, session):
        window = LiamWindow(self.get_application(), self.model, self.auto_confirm)
        window.show_all()
        # Same show_all()-overrides-set_visible(False) issue as
        # LiamApp.do_activate — re-hide the artifacts panel here too.
        window.artifacts_box.set_visible(window.artifacts_toggle.get_active())
        window.restore_saved_placement()
        window.present()
        window._switch_to(session["folder_path"], session["title"])

    def _open_in_file_manager(self, session):
        Gtk.show_uri_on_window(self, f"file://{session['folder_path']}", Gdk.CURRENT_TIME)

    def _toggle_pin(self, session):
        memory.set_pinned(session["id"], not session.get("pinned"))
        self._refresh_session_list()
        self._select_row_for_session(session["id"])

    def _toggle_unread(self, session):
        memory.set_unread(session["id"], not session.get("unread"))
        self._refresh_session_list()
        self._select_row_for_session(session["id"])

    def _fork_session(self, session):
        new_id = memory.fork_session(session["id"])
        if new_id is None:
            return
        self._refresh_session_list()
        self._select_row_for_session(new_id)

    def _add_session_folder(self, session):
        # A non-modal dialog (connect("response",...) + show_all(), never
        # .run()) — Gtk.Dialog.run() forces modal=True for as long as it
        # blocks, regardless of what's set beforehand, and GNOME's
        # attach-modal-dialogs (on by default here) visually glues any
        # modal dialog to its parent so they drag as one unit. Confirmed
        # by direct test, not assumed — every dialog in this file needs
        # the same non-blocking treatment for that reason.
        dialog = Gtk.FileChooserDialog(
            title="Choose a folder to add to this thread",
            parent=self,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
        )
        dialog.set_position(Gtk.WindowPosition.CENTER)
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK,
        )

        def on_response(dialog, response):
            path = dialog.get_filename() if response == Gtk.ResponseType.OK else None
            dialog.destroy()
            if not path:
                return
            memory.add_session_folder(session["id"], path)
            self._refresh_session_list()
            self._select_row_for_session(session["id"])
            if session["id"] == self.session_id:
                # Rebuild the live agent in place so the new folder is
                # usable right away — reuses the existing chat buffer/
                # history as-is, unlike _switch_to (for changing which
                # thread is open).
                self._reload_current_agent()

        dialog.connect("response", on_response)
        dialog.show_all()

    def _build_group_submenu(self, session):
        submenu = Gtk.Menu()

        if session.get("group_id") is not None:
            none_item = Gtk.MenuItem(label="No group")
            none_item.connect("activate", lambda _item: self._set_session_group(session, None))
            submenu.append(none_item)
            submenu.append(Gtk.SeparatorMenuItem())

        for group in memory.list_groups():
            item = Gtk.MenuItem(label=group["name"])
            item.set_sensitive(group["id"] != session.get("group_id"))
            item.connect("activate", lambda _item, g=group: self._set_session_group(session, g["id"]))
            submenu.append(item)

        submenu.append(Gtk.SeparatorMenuItem())
        new_group_item = Gtk.MenuItem(label="New group…")
        new_group_item.connect("activate", lambda _item: self._prompt_new_group(session))
        submenu.append(new_group_item)

        submenu.show_all()
        return submenu

    def _set_session_group(self, session, group_id):
        memory.set_session_group(session["id"], group_id)
        self._refresh_session_list()
        self._select_row_for_session(session["id"])

    def _prompt_new_group(self, session):
        dialog = Gtk.Dialog(title="New Group", transient_for=self)
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK, Gtk.ResponseType.OK,
        )
        dialog.set_default_response(Gtk.ResponseType.OK)

        entry = Gtk.Entry()
        entry.set_placeholder_text("Group name")
        entry.set_activates_default(True)
        content = dialog.get_content_area()
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        content.pack_start(entry, False, False, 0)

        def on_response(dialog, response):
            name = entry.get_text().strip()
            dialog.destroy()
            if response == Gtk.ResponseType.OK and name:
                group_id = memory.create_group(name)
                self._set_session_group(session, group_id)

        dialog.connect("response", on_response)
        dialog.show_all()

    def _archive_session(self, session):
        memory.set_archived(session["id"], True)
        was_active = session["id"] == self.session_id
        self._refresh_session_list()
        if not was_active:
            return
        if not self.sessions:
            memory.get_or_create_session(os.getcwd())
            self._refresh_session_list()
        next_session = self.sessions[0]
        self._switch_to(next_session["folder_path"], next_session["title"])
        self._select_row_for_session(next_session["id"])

    def _rename_session(self, session):
        dialog = Gtk.Dialog(title="Rename Thread", transient_for=self)
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK, Gtk.ResponseType.OK,
        )
        dialog.set_default_response(Gtk.ResponseType.OK)

        entry = Gtk.Entry()
        entry.set_text(session["title"])
        entry.set_activates_default(True)
        content = dialog.get_content_area()
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        content.pack_start(entry, False, False, 0)

        def on_response(dialog, response):
            new_title = entry.get_text().strip()
            dialog.destroy()
            if response != Gtk.ResponseType.OK or not new_title or new_title == session["title"]:
                return
            memory.rename_session(session["id"], new_title)
            self._refresh_session_list()
            self._select_row_for_session(session["id"])
            if session["id"] == self.session_id:
                self.headerbar.set_title(new_title)

        dialog.connect("response", on_response)
        dialog.show_all()

    def _delete_session(self, session):
        if self.busy:
            return
        confirm = Gtk.MessageDialog(
            transient_for=self,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.NONE,
            text=f'Delete "{session["title"]}"?',
            secondary_text="This removes its entire message history. This can't be undone.",
        )
        confirm.add_buttons(
            "Cancel", Gtk.ResponseType.CANCEL,
            "Delete", Gtk.ResponseType.YES,
        )

        def on_response(confirm, response):
            confirm.destroy()
            if response != Gtk.ResponseType.YES:
                return

            memory.delete_session(session["id"])
            was_active = session["id"] == self.session_id
            self._refresh_session_list()

            if not was_active:
                return

            # The deleted thread was the one on screen — switch to
            # whatever's next, or bootstrap a fresh default if nothing's
            # left.
            if not self.sessions:
                memory.get_or_create_session(os.getcwd())
                self._refresh_session_list()
            next_session = self.sessions[0]
            self._switch_to(next_session["folder_path"], next_session["title"])
            self._select_row_for_session(next_session["id"])

        confirm.connect("response", on_response)
        confirm.show_all()

    # --- customize (settings) ---

    def _list_ollama_models(self):
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().splitlines()[1:]  # skip the header row
            return [line.split()[0] for line in lines if line.strip()]
        except Exception:
            return []

    def _open_settings_dialog(self, _button):
        dialog = Gtk.Dialog(title="Customize", transient_for=self)
        dialog.set_position(Gtk.WindowPosition.CENTER)
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_SAVE, Gtk.ResponseType.OK,
        )
        dialog.set_default_response(Gtk.ResponseType.OK)

        content = dialog.get_content_area()
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        content.set_spacing(8)
        dialog.set_default_size(620, 520)

        content.pack_start(Gtk.Label(label="Model", xalign=0), False, False, 0)
        model_combo = Gtk.ComboBoxText()
        current_model = self.settings["model"] or DEFAULT_MODEL
        models = self._list_ollama_models()
        if current_model not in models:
            models = [current_model] + models
        for name in models:
            model_combo.append(name, name)
        model_combo.set_active_id(current_model)
        content.pack_start(model_combo, False, False, 0)

        auto_confirm_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        auto_confirm_row.pack_start(
            Gtk.Label(label="Run tools without asking (write_file, shell commands)", xalign=0),
            True, True, 0,
        )
        auto_confirm_switch = Gtk.Switch()
        auto_confirm_switch.get_style_context().add_class("liam-toggle")
        auto_confirm_switch.set_active(self.settings["auto_confirm"])
        auto_confirm_row.pack_start(auto_confirm_switch, False, False, 0)
        content.pack_start(auto_confirm_row, False, False, 0)

        sudo_expander = Gtk.Expander(label="SSH sudo passwords (GNOME Keyring)")
        sudo_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        sudo_password_rows = []
        sudo_box.set_margin_top(8)
        sudo_box.set_margin_start(8)
        sudo_box.set_margin_end(8)
        sudo_box.pack_start(
            Gtk.Label(
                label=(
                    "Optional credentials for allowlisted desktop SSH hosts. "
                    "Passwords are never shown to Liam. Save/replace and remove "
                    "actions take effect immediately."
                ),
                xalign=0,
                wrap=True,
            ),
            False, False, 0,
        )

        configured_hosts = agent_tools._configured_ssh_hosts()
        if not configured_hosts:
            empty = Gtk.Label(
                label="No hosts are configured in LIAM_SSH_HOSTS.", xalign=0,
            )
            empty.get_style_context().add_class("dim-label")
            sudo_box.pack_start(empty, False, False, 0)

        for alias in configured_hosts:
            try:
                identity = agent_tools._ssh_host_details(alias)
            except (OSError, ValueError, subprocess.SubprocessError):
                error = Gtk.Label(
                    label=f"{alias}: SSH configuration could not be resolved.",
                    xalign=0,
                )
                error.get_style_context().add_class("error")
                sudo_box.pack_start(error, False, False, 0)
                continue

            host_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            host_box.pack_start(
                Gtk.Label(
                    label=(
                        f"{alias} — {identity['user']}@{identity['hostname']}:"
                        f"{identity['port']}"
                    ),
                    xalign=0,
                ),
                False, False, 0,
            )
            controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            password_entry = Gtk.Entry()
            password_entry.set_visibility(False)
            password_entry.set_input_purpose(Gtk.InputPurpose.PASSWORD)
            password_entry.set_placeholder_text("Sudo password")
            controls.pack_start(password_entry, True, True, 0)
            save_password = Gtk.Button(label=f"Save for {alias}")
            remove_password = Gtk.Button(label=f"Remove from {alias}")
            controls.pack_start(save_password, False, False, 0)
            controls.pack_start(remove_password, False, False, 0)
            host_box.pack_start(controls, False, False, 0)
            credential_status = Gtk.Label(xalign=0)
            credential_status.get_style_context().add_class("dim-label")
            host_box.pack_start(credential_status, False, False, 0)

            def set_initial_status(identity=identity, label=credential_status):
                try:
                    stored = ssh_secrets.has_sudo_password(
                        identity["alias"], identity["hostname"],
                        identity["port"], identity["user"],
                    )
                    label.set_text(
                        f"Password stored for {identity['alias']}"
                        if stored else f"No password stored for {identity['alias']}"
                    )
                except ssh_secrets.SudoSecretError as exc:
                    label.set_text(str(exc))

            def on_save_password(
                _button, identity=identity, entry=password_entry,
                label=credential_status,
            ):
                password = entry.get_text()
                try:
                    ssh_secrets.store_sudo_password(
                        identity["alias"], identity["hostname"],
                        identity["port"], identity["user"], password,
                    )
                except ssh_secrets.SudoSecretError as exc:
                    label.set_text(str(exc))
                    return
                entry.set_text("")
                label.set_text(
                    f"Password saved for {identity['alias']} in GNOME Keyring"
                )

            def on_remove_password(
                _button, identity=identity, entry=password_entry,
                label=credential_status,
            ):
                try:
                    removed = ssh_secrets.clear_sudo_password(
                        identity["alias"], identity["hostname"],
                        identity["port"], identity["user"],
                    )
                except ssh_secrets.SudoSecretError as exc:
                    label.set_text(str(exc))
                    return
                entry.set_text("")
                label.set_text(
                    f"Password removed from {identity['alias']}"
                    if removed else f"No password was stored for {identity['alias']}"
                )

            save_password.connect("clicked", on_save_password)
            remove_password.connect("clicked", on_remove_password)
            set_initial_status()
            sudo_password_rows.append((identity, password_entry, credential_status))
            sudo_box.pack_start(host_box, False, False, 0)

        sudo_expander.add(sudo_box)
        content.pack_start(sudo_expander, False, False, 0)

        content.pack_start(
            Gtk.Label(label="Custom instructions (added to every thread's system prompt)", xalign=0),
            False, False, 0,
        )
        instructions_buffer = Gtk.TextBuffer()
        instructions_buffer.set_text(self.settings["custom_instructions"])
        instructions_view = Gtk.TextView(buffer=instructions_buffer)
        instructions_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        instructions_scroller = Gtk.ScrolledWindow()
        instructions_scroller.set_vexpand(True)
        instructions_scroller.add(instructions_view)
        content.pack_start(instructions_scroller, True, True, 0)

        def on_response(dialog, response):
            if response == Gtk.ResponseType.OK:
                # The dialog-level Save button must honor passwords typed into
                # host rows. Previously only the small per-row button stored
                # them, so a user could enter a password, click the prominent
                # Save button, and reasonably believe it had been committed.
                for identity, entry, label in sudo_password_rows:
                    password = entry.get_text()
                    if not password:
                        continue
                    try:
                        ssh_secrets.store_sudo_password(
                            identity["alias"], identity["hostname"],
                            identity["port"], identity["user"], password,
                        )
                    except ssh_secrets.SudoSecretError as exc:
                        label.set_text(str(exc))
                        sudo_expander.set_expanded(True)
                        return
                    entry.set_text("")
                    label.set_text(
                        f"Password saved for {identity['alias']} in GNOME Keyring"
                    )
                start, end = instructions_buffer.get_bounds()
                self.settings["model"] = model_combo.get_active_id()
                self.settings["auto_confirm"] = auto_confirm_switch.get_active()
                self.settings["custom_instructions"] = instructions_buffer.get_text(start, end, False)
                liam_settings.save(self.settings)
                self.model = self.settings["model"] or DEFAULT_MODEL
                self.auto_confirm = self.settings["auto_confirm"]
                self._reload_current_agent()
            dialog.destroy()

        dialog.connect("response", on_response)
        dialog.show_all()

    def _reload_current_agent(self):
        """Rebuild the live agent in place with current settings/folders
        — reuses the existing chat buffer/history as-is, the same pattern
        _add_session_folder already uses, rather than _switch_to (which
        is for changing which thread is open, and would wipe/replay it)."""
        if self.session_id is None:
            return
        session = memory.get_session(self.session_id)
        if session is None:
            return
        agent, _history = self._build_agent_and_history(session["id"], session["folder_path"])
        self.agent = agent
        agent.on_tool_call = self._on_tool_call
        agent.on_status = self._on_status
        agent.on_confirm = self._on_confirm

    # --- lessons ---

    def _open_lessons_dialog(self, _button):
        dialog = Gtk.Dialog(title="Lessons", transient_for=self)
        dialog.set_position(Gtk.WindowPosition.CENTER)
        dialog.set_default_size(820, 620)
        dialog.add_button(Gtk.STOCK_CLOSE, Gtk.ResponseType.CLOSE)

        content = dialog.get_content_area()
        content.set_margin_top(10)
        content.set_margin_bottom(10)
        content.set_margin_start(10)
        content.set_margin_end(10)
        content.set_spacing(8)

        filter_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        filter_row.pack_start(Gtk.Label(label="View", xalign=0), False, False, 0)
        status_combo = Gtk.ComboBoxText()
        for status, label in (
            ("pending", "Pending"),
            ("active", "Active"),
            ("disabled", "Disabled"),
            ("quarantined", "Quarantined"),
        ):
            status_combo.append(status, label)
        status_combo.set_active_id("pending")
        filter_row.pack_start(status_combo, False, False, 0)
        refresh_button = Gtk.Button.new_from_icon_name("view-refresh-symbolic", Gtk.IconSize.BUTTON)
        refresh_button.set_tooltip_text("Refresh lessons now")
        filter_row.pack_end(refresh_button, False, False, 0)
        auto_refresh_label = Gtk.Label(label="Auto-refreshes every 3 seconds")
        auto_refresh_label.get_style_context().add_class("dim-label")
        auto_refresh_label.set_tooltip_text("Automatic refresh pauses while you have unsaved edits.")
        filter_row.pack_end(auto_refresh_label, False, False, 0)
        content.pack_start(filter_row, False, False, 0)

        main = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        main.set_position(290)
        content.pack_start(main, True, True, 0)

        lesson_list = Gtk.ListBox()
        lesson_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        lesson_list.get_style_context().add_class("liam-select-list")
        empty_label = Gtk.Label(label="No lessons in this view.")
        empty_label.get_style_context().add_class("dim-label")
        empty_label.set_margin_top(16)
        empty_label.set_margin_bottom(16)
        lesson_list.set_placeholder(empty_label)
        list_scroller = Gtk.ScrolledWindow()
        list_scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        list_scroller.add(lesson_list)
        main.pack1(list_scroller, resize=False, shrink=False)

        editor = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=7)
        editor.set_margin_start(10)
        main.pack2(editor, resize=True, shrink=False)

        provenance_label = Gtk.Label(xalign=0)
        provenance_label.set_line_wrap(True)
        provenance_label.get_style_context().add_class("dim-label")
        editor.pack_start(provenance_label, False, False, 0)

        editor.pack_start(Gtk.Label(label="Trigger keywords or phrases", xalign=0), False, False, 0)
        keywords_entry = Gtk.Entry()
        editor.pack_start(keywords_entry, False, False, 0)

        editor.pack_start(Gtk.Label(label="Correct behavior", xalign=0), False, False, 0)
        lesson_buffer = Gtk.TextBuffer()
        lesson_view = Gtk.TextView(buffer=lesson_buffer)
        lesson_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        lesson_scroller = Gtk.ScrolledWindow()
        lesson_scroller.set_min_content_height(90)
        lesson_scroller.add(lesson_view)
        editor.pack_start(lesson_scroller, False, False, 0)

        scope_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=7)
        scope_row.pack_start(Gtk.Label(label="Scope", xalign=0), False, False, 0)
        scope_combo = Gtk.ComboBoxText()
        for scope in ("global", "workspace", "channel", "tool"):
            scope_combo.append(scope, scope.capitalize())
        scope_row.pack_start(scope_combo, False, False, 0)
        scope_value_entry = Gtk.Entry()
        scope_value_entry.set_placeholder_text("Workspace path, channel, or tool")
        scope_row.pack_start(scope_value_entry, True, True, 0)
        editor.pack_start(scope_row, False, False, 0)

        stats_label = Gtk.Label(xalign=0)
        stats_label.get_style_context().add_class("dim-label")
        editor.pack_start(stats_label, False, False, 0)

        editor.pack_start(Gtk.Label(label="Evidence and provenance", xalign=0), False, False, 0)
        evidence_buffer = Gtk.TextBuffer()
        evidence_view = Gtk.TextView(buffer=evidence_buffer)
        evidence_view.set_editable(False)
        evidence_view.set_cursor_visible(False)
        evidence_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        evidence_scroller = Gtk.ScrolledWindow()
        evidence_scroller.set_vexpand(True)
        evidence_scroller.add(evidence_view)
        editor.pack_start(evidence_scroller, True, True, 0)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        save_button = Gtk.Button(label="Approve and activate")
        toggle_button = Gtk.Button(label="Disable")
        merge_button = Gtk.Button(label="Merge…")
        reject_button = Gtk.Button(label="Reject")
        delete_button = Gtk.Button(label="Delete")
        actions.pack_start(save_button, False, False, 0)
        actions.pack_start(toggle_button, False, False, 0)
        actions.pack_start(merge_button, False, False, 0)
        actions.pack_end(delete_button, False, False, 0)
        actions.pack_end(reject_button, False, False, 0)
        editor.pack_start(actions, False, False, 0)

        state = {
            "lesson": None,
            "loading": False,
            "dirty": False,
            "signature": None,
            "refresh_source": 0,
            "closed": False,
        }

        def show_error(message):
            error = Gtk.MessageDialog(
                transient_for=dialog,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.CLOSE,
                text="Could not update lesson",
                secondary_text=str(message),
            )
            error.connect("response", lambda widget, _response: widget.destroy())
            error.show_all()

        def set_editor_sensitive(enabled):
            for widget in (
                keywords_entry, lesson_view, scope_combo, scope_value_entry,
                save_button, toggle_button, merge_button, reject_button, delete_button,
            ):
                widget.set_sensitive(enabled)

        def populate_editor(record):
            state["loading"] = True
            try:
                state["lesson"] = record
                if record is None:
                    provenance_label.set_text("Select a lesson to inspect or edit it.")
                    keywords_entry.set_text("")
                    lesson_buffer.set_text("")
                    scope_combo.set_active_id("global")
                    scope_value_entry.set_text("")
                    stats_label.set_text("")
                    evidence_buffer.set_text("")
                    set_editor_sensitive(False)
                    return
                set_editor_sensitive(True)
                provenance_label.set_text(
                    f"#{record['id']} · {record['origin']} · {record.get('detector') or 'no detector'}"
                )
                keywords_entry.set_text(record["keywords"])
                lesson_buffer.set_text(record["lesson"])
                scope_combo.set_active_id(record["scope_kind"])
                scope_value_entry.set_text(record.get("scope_value") or "")
                scope_value_entry.set_sensitive(record["scope_kind"] != "global")
                stats_label.set_text(
                    f"Observed {record['occurrence_count']} · used {record['hit_count']} · "
                    f"verified success {record['success_count']} · failure {record['failure_count']}"
                )
                events = memory.list_lesson_events(record["id"], limit=12)
                event_text = []
                for event in events:
                    source = " / ".join(
                        value for value in (
                            event.get("source_channel"), event.get("source_actor")
                        ) if value
                    ) or "system"
                    event_text.append(
                        f"{event['created_at']} · {event['event_kind']} · {source}\n"
                        f"{event.get('evidence') or '(no excerpt retained)'}"
                    )
                evidence_buffer.set_text("\n\n———\n\n".join(event_text))
                status = record["status"]
                save_button.set_label(
                    "Save" if status == "active" else "Approve and activate"
                )
                toggle_button.set_label("Disable" if status == "active" else "Reactivate")
                toggle_button.set_sensitive(status in {"active", "disabled", "quarantined"})
                reject_button.set_sensitive(status in {"pending", "quarantined"})
            finally:
                state["loading"] = False
                state["dirty"] = False

        def refresh(preferred_id=None, force=False):
            if state["dirty"] and not force:
                return
            status = status_combo.get_active_id() or "pending"
            records = memory.list_lessons(status=status)
            signature = (
                status,
                tuple(
                    (
                        record["id"], record["status"], str(record.get("updated_at")),
                        record["occurrence_count"], record["hit_count"],
                        record["success_count"], record["failure_count"],
                        record["keywords"], record["lesson"], record["scope_kind"],
                        record.get("scope_value"),
                    )
                    for record in records
                ),
            )
            if not force and signature == state["signature"]:
                return
            state["signature"] = signature
            if preferred_id is None and state["lesson"] is not None:
                preferred_id = state["lesson"]["id"]
            for child in list(lesson_list.get_children()):
                lesson_list.remove(child)
            selected_row = None
            for record in records:
                box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
                box.set_margin_top(6)
                box.set_margin_bottom(6)
                box.set_margin_start(8)
                box.set_margin_end(8)
                title = record["lesson"] or "(redacted)"
                title_label = Gtk.Label(label=f"#{record['id']}  {title}", xalign=0)
                title_label.set_ellipsize(Pango.EllipsizeMode.END)
                subtitle = Gtk.Label(
                    label=f"{record['origin']} · {record['scope_kind']}", xalign=0,
                )
                subtitle.get_style_context().add_class("dim-label")
                box.pack_start(title_label, False, False, 0)
                box.pack_start(subtitle, False, False, 0)
                row = Gtk.ListBoxRow()
                row.lesson_data = record
                row.add(box)
                lesson_list.add(row)
                if preferred_id == record["id"]:
                    selected_row = row
            lesson_list.show_all()
            rows = lesson_list.get_children()
            selected_row = selected_row or (rows[0] if rows else None)
            if selected_row:
                lesson_list.select_row(selected_row)
            else:
                populate_editor(None)

        def on_selected(_listbox, row):
            populate_editor(getattr(row, "lesson_data", None) if row else None)

        def on_scope_changed(_combo):
            scope_value_entry.set_sensitive(scope_combo.get_active_id() != "global")
            if not state["loading"] and state["lesson"] is not None:
                state["dirty"] = True

        def on_editor_changed(*_args):
            if not state["loading"] and state["lesson"] is not None:
                state["dirty"] = True

        def on_save(_widget):
            record = state["lesson"]
            if record is None:
                return
            start, end = lesson_buffer.get_bounds()
            next_status = "active"
            try:
                updated = memory.update_lesson(
                    record["id"], keywords=keywords_entry.get_text(),
                    lesson=lesson_buffer.get_text(start, end, False),
                    status=next_status,
                    scope_kind=scope_combo.get_active_id(),
                    scope_value=scope_value_entry.get_text(),
                )
                if updated is None:
                    raise RuntimeError("The database did not update the lesson.")
                state["dirty"] = False
                status_combo.set_active_id("active")
                refresh(updated["id"], force=True)
            except Exception as exc:
                show_error(exc)

        def on_toggle(_widget):
            record = state["lesson"]
            if record is None:
                return
            next_status = "disabled" if record["status"] == "active" else "active"
            updated = memory.update_lesson(record["id"], status=next_status)
            if updated is None:
                show_error("The database did not update the lesson.")
                return
            state["dirty"] = False
            status_combo.set_active_id(next_status)
            refresh(updated["id"], force=True)

        def on_reject(_widget):
            record = state["lesson"]
            if record and memory.reject_lesson(record["id"]):
                refresh()

        def on_delete(_widget):
            record = state["lesson"]
            if record is None:
                return
            confirm = Gtk.MessageDialog(
                transient_for=dialog,
                message_type=Gtk.MessageType.WARNING,
                buttons=Gtk.ButtonsType.NONE,
                text=f"Delete lesson #{record['id']}?",
                secondary_text="This also removes its provenance and effectiveness history.",
            )
            confirm.add_buttons("Cancel", Gtk.ResponseType.CANCEL, "Delete", Gtk.ResponseType.YES)

            def on_response(widget, response):
                widget.destroy()
                if response == Gtk.ResponseType.YES:
                    memory.delete_lesson(record["id"])
                    refresh()

            confirm.connect("response", on_response)
            confirm.show_all()

        def on_merge(_widget):
            record = state["lesson"]
            if record is None:
                return
            candidates = [
                item for item in memory.list_lessons()
                if item["id"] != record["id"]
            ]
            if not candidates:
                show_error("There are no other lessons to merge into.")
                return
            merge = Gtk.Dialog(title="Merge lesson", transient_for=dialog)
            merge.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, "Merge", Gtk.ResponseType.OK)
            box = merge.get_content_area()
            box.set_margin_top(10)
            box.set_margin_bottom(10)
            box.set_margin_start(10)
            box.set_margin_end(10)
            box.set_spacing(6)
            box.add(Gtk.Label(label=f"Merge lesson #{record['id']} into:", xalign=0))
            target_combo = Gtk.ComboBoxText()
            for candidate in candidates:
                target_combo.append(str(candidate["id"]), f"#{candidate['id']}  {candidate['lesson'][:80]}")
            target_combo.set_active(0)
            box.add(target_combo)

            def on_response(widget, response):
                target_id = target_combo.get_active_id()
                widget.destroy()
                if response == Gtk.ResponseType.OK and target_id:
                    if memory.merge_lessons(record["id"], int(target_id)):
                        refresh()
                    else:
                        show_error("The lessons could not be merged.")

            merge.connect("response", on_response)
            merge.show_all()

        lesson_list.connect("row-selected", on_selected)
        status_combo.connect("changed", lambda _combo: refresh(force=True))
        scope_combo.connect("changed", on_scope_changed)
        keywords_entry.connect("changed", on_editor_changed)
        lesson_buffer.connect("changed", on_editor_changed)
        scope_value_entry.connect("changed", on_editor_changed)
        refresh_button.connect("clicked", lambda _button: refresh(force=True))
        save_button.connect("clicked", on_save)
        toggle_button.connect("clicked", on_toggle)
        reject_button.connect("clicked", on_reject)
        delete_button.connect("clicked", on_delete)
        merge_button.connect("clicked", on_merge)
        dialog.connect("response", lambda widget, _response: widget.destroy())

        def poll_refresh():
            if state["closed"]:
                state["refresh_source"] = 0
                return GLib.SOURCE_REMOVE
            refresh()
            return GLib.SOURCE_CONTINUE

        def on_destroy(_widget):
            state["closed"] = True
            source_id = state["refresh_source"]
            state["refresh_source"] = 0
            if source_id:
                GLib.source_remove(source_id)

        dialog.connect("destroy", on_destroy)

        refresh(force=True)
        dialog.show_all()
        # show_all() re-enables the value field even for global scope.
        state["loading"] = True
        on_scope_changed(scope_combo)
        state["loading"] = False
        state["refresh_source"] = GLib.timeout_add_seconds(3, poll_refresh)

    # --- routines ---

    def _configure_dialog_geometry(self, dialog, key, default_width, default_height):
        """Restore a resizable dialog without trusting stale monitor geometry."""
        all_geometry = self.settings.get("dialog_geometry")
        saved = all_geometry.get(key, {}) if isinstance(all_geometry, dict) else {}
        try:
            width = max(320, int(saved.get("width", default_width)))
            height = max(240, int(saved.get("height", default_height)))
            x = int(saved["x"])
            y = int(saved["y"])
        except (KeyError, TypeError, ValueError):
            dialog.set_position(Gtk.WindowPosition.CENTER)
            dialog.set_default_size(default_width, default_height)
            return

        display = self.get_display()
        monitor_index = _monitor_index_at_point(
            display, x + width // 2, y + height // 2,
        )
        monitor = display.get_monitor(monitor_index) if monitor_index >= 0 else None
        if monitor is None:
            monitor = display.get_primary_monitor()
        if monitor is None and display.get_n_monitors() > 0:
            monitor = display.get_monitor(0)
        if monitor is not None:
            workarea = monitor.get_workarea()
            width = min(width, workarea.width)
            height = min(height, workarea.height)
            x = max(workarea.x, min(x, workarea.x + workarea.width - width))
            y = max(workarea.y, min(y, workarea.y + workarea.height - height))

        dialog.set_default_size(width, height)
        dialog.resize(width, height)
        _move_window(dialog, x, y)

        def restore_after_map(_widget, _event):
            # Mutter can ignore pre-map positioning, so repeat it once the
            # dialog owns a real window, just as the main window does.
            GLib.idle_add(lambda: (_move_window(dialog, x, y), False)[1])
            return False

        dialog.connect("map-event", restore_after_map)

    def _save_dialog_geometry(self, dialog, key):
        gdk_window = dialog.get_window()
        if gdk_window is None:
            return
        frame = gdk_window.get_frame_extents()
        width, height = dialog.get_size()
        all_geometry = self.settings.get("dialog_geometry")
        all_geometry = dict(all_geometry) if isinstance(all_geometry, dict) else {}
        all_geometry[key] = {
            "x": frame.x,
            "y": frame.y,
            "width": width,
            "height": height,
        }
        self.settings["dialog_geometry"] = all_geometry
        liam_settings.save(self.settings)

    def _open_routines_dialog(self, _button):
        dialog = Gtk.Dialog(title="Routines", transient_for=self)
        self._configure_dialog_geometry(dialog, "routines", 520, 400)
        dialog.add_button(Gtk.STOCK_CLOSE, Gtk.ResponseType.CLOSE)

        content = dialog.get_content_area()
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        content.set_spacing(8)

        routines_list = Gtk.ListBox()
        # Each routine row contains its own controls and has no row-level
        # action, so selecting the entire row is misleading.
        routines_list.set_selection_mode(Gtk.SelectionMode.NONE)
        routines_list.get_style_context().add_class("liam-select-list")
        list_scroller = Gtk.ScrolledWindow()
        list_scroller.set_vexpand(True)
        list_scroller.add(routines_list)
        content.pack_start(list_scroller, True, True, 0)

        new_button = Gtk.Button(label="New routine…")
        new_button.connect("clicked", lambda _b: self._prompt_new_routine(dialog, routines_list))
        content.pack_start(new_button, False, False, 0)

        self._refresh_routines_list(routines_list)

        def close_dialog(widget, _response):
            self._save_dialog_geometry(widget, "routines")
            widget.destroy()

        dialog.connect("response", close_dialog)
        dialog.show_all()

    def _refresh_routines_list(self, routines_list):
        for child in list(routines_list.get_children()):
            routines_list.remove(child)
        sessions_by_id = {s["id"]: s for s in memory.list_sessions(include_archived=True)}
        for routine in routines.list_routines():
            session = sessions_by_id.get(routine["session_id"])
            thread_name = session["title"] if session else "(deleted thread)"
            if routine["schedule_kind"] == "once":
                schedule_text = f"once at {routine['schedule_value']}"
            elif routine["schedule_kind"] == "daily":
                schedule_text = f"daily at {routine['schedule_value']}"
            elif routine["schedule_kind"] == "minutely":
                schedule_text = f"every {routine['schedule_value']}m"
            else:
                schedule_text = f"every {routine['schedule_value']}h"
            finished_status = _finished_once_status(routine)
            last_run = routine["last_run_at"] or "never"
            if finished_status == "Completed":
                run_text = f"completed {last_run}"
            elif finished_status:
                run_text = finished_status.lower()
            else:
                run_text = f"last ran {last_run}"
            display_prompt = _routine_display_prompt(routine["prompt"])
            prompt_preview = display_prompt[:60] + ("…" if len(display_prompt) > 60 else "")

            row_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            row_box.set_margin_top(6)
            row_box.set_margin_bottom(6)
            row_box.set_margin_start(8)
            row_box.set_margin_end(8)
            row_box.pack_start(Gtk.Label(label=prompt_preview, xalign=0), False, False, 0)
            detail_label = Gtk.Label(
                label=f"{thread_name} · {schedule_text} · {run_text}", xalign=0,
            )
            detail_label.get_style_context().add_class("dim-label")
            row_box.pack_start(detail_label, False, False, 0)

            controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            if finished_status:
                status_label = Gtk.Label(label=finished_status)
                status_label.get_style_context().add_class("dim-label")
                if finished_status == "Completed":
                    status_label.set_tooltip_text(
                        "This one-time routine already ran. Create a new routine to run it again."
                    )
                elif finished_status == "Expired":
                    status_label.set_tooltip_text(
                        "This one-time schedule is in the past. Create a new routine with a future time."
                    )
                controls.pack_start(status_label, False, False, 0)
            else:
                enabled_switch = Gtk.Switch()
                enabled_switch.get_style_context().add_class("liam-toggle")
                enabled_switch.set_active(routine["enabled"])

                def on_state_set(_switch, active, rid=routine["id"]):
                    try:
                        routines.set_enabled(rid, active)
                    except Exception as exc:
                        error = Gtk.MessageDialog(
                            transient_for=self,
                            message_type=Gtk.MessageType.ERROR,
                            buttons=Gtk.ButtonsType.CLOSE,
                            text="Could not update routine",
                            secondary_text=str(exc),
                        )
                        error.connect("response", lambda widget, _response: widget.destroy())
                        error.show_all()
                        GLib.idle_add(self._refresh_routines_list, routines_list)
                        return True
                    return False

                enabled_switch.connect("state-set", on_state_set)
                controls.pack_start(enabled_switch, False, False, 0)
            delete_button = Gtk.Button(label="Delete")
            delete_button.connect(
                "clicked",
                lambda _b, rid=routine["id"]: (routines.delete_routine(rid), self._refresh_routines_list(routines_list)),
            )
            controls.pack_start(delete_button, False, False, 0)
            row_box.pack_start(controls, False, False, 0)

            row = Gtk.ListBoxRow()
            row.add(row_box)
            routines_list.add(row)
        routines_list.show_all()

    def _prompt_new_routine(self, _parent_dialog, routines_list):
        dialog = Gtk.Dialog(title="New Routine", transient_for=self)
        dialog.set_position(Gtk.WindowPosition.CENTER)
        dialog.set_default_size(420, 320)
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK, Gtk.ResponseType.OK,
        )
        dialog.set_default_response(Gtk.ResponseType.OK)

        content = dialog.get_content_area()
        content.set_margin_top(12)
        content.set_margin_bottom(12)
        content.set_margin_start(12)
        content.set_margin_end(12)
        content.set_spacing(8)

        content.pack_start(Gtk.Label(label="Thread", xalign=0), False, False, 0)
        thread_combo = Gtk.ComboBoxText()
        sessions = memory.list_sessions()
        for session in sessions:
            thread_combo.append(str(session["id"]), session["title"])
        if sessions:
            thread_combo.set_active(0)
        content.pack_start(thread_combo, False, False, 0)

        content.pack_start(Gtk.Label(label="Prompt", xalign=0), False, False, 0)
        prompt_buffer = Gtk.TextBuffer()
        prompt_view = Gtk.TextView(buffer=prompt_buffer)
        prompt_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        prompt_scroller = Gtk.ScrolledWindow()
        prompt_scroller.set_vexpand(True)
        prompt_scroller.add(prompt_view)
        content.pack_start(prompt_scroller, True, True, 0)

        content.pack_start(Gtk.Label(label="Schedule", xalign=0), False, False, 0)
        schedule_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        kind_combo = Gtk.ComboBoxText()
        kind_combo.append("once", "Once at")
        kind_combo.append("daily", "Daily at")
        kind_combo.append("minutely", "Every N minutes")
        kind_combo.append("hourly", "Every N hours")
        kind_combo.set_active_id("daily")
        schedule_row.pack_start(kind_combo, False, False, 0)

        time_entry = Gtk.Entry()
        time_entry.set_text("08:00")
        time_entry.set_placeholder_text("HH:MM")
        time_entry.set_width_chars(6)
        schedule_row.pack_start(time_entry, False, False, 0)

        once_entry = Gtk.Entry()
        once_entry.set_text(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + 3600)))
        once_entry.set_placeholder_text("YYYY-MM-DD HH:MM:SS")
        once_entry.set_width_chars(19)
        once_entry.set_visible(False)
        schedule_row.pack_start(once_entry, False, False, 0)

        minutes_spin = Gtk.SpinButton.new_with_range(1, 1440, 1)
        minutes_spin.set_value(5)
        minutes_spin.set_visible(False)
        schedule_row.pack_start(minutes_spin, False, False, 0)

        hours_spin = Gtk.SpinButton.new_with_range(1, 168, 1)
        hours_spin.set_value(4)
        hours_spin.set_visible(False)
        schedule_row.pack_start(hours_spin, False, False, 0)

        def on_kind_changed(_combo):
            kind = kind_combo.get_active_id()
            is_daily = kind == "daily"
            time_entry.set_visible(is_daily)
            once_entry.set_visible(kind == "once")
            minutes_spin.set_visible(kind == "minutely")
            hours_spin.set_visible(kind == "hourly")

        kind_combo.connect("changed", on_kind_changed)
        content.pack_start(schedule_row, False, False, 0)

        def on_response(dialog, response):
            if response == Gtk.ResponseType.OK and sessions and thread_combo.get_active_id():
                start, end = prompt_buffer.get_bounds()
                prompt = prompt_buffer.get_text(start, end, False).strip()
                schedule_kind = kind_combo.get_active_id()
                if schedule_kind == "once":
                    schedule_value = once_entry.get_text().strip()
                elif schedule_kind == "daily":
                    schedule_value = time_entry.get_text().strip()
                elif schedule_kind == "minutely":
                    schedule_value = str(int(minutes_spin.get_value()))
                else:
                    schedule_value = str(int(hours_spin.get_value()))
                if prompt:
                    routines.create_routine(int(thread_combo.get_active_id()), prompt, schedule_kind, schedule_value)
                    self._refresh_routines_list(routines_list)
            dialog.destroy()

        dialog.connect("response", on_response)
        dialog.show_all()
        on_kind_changed(kind_combo)

    # --- artifacts ---

    def _on_artifacts_toggled(self, button):
        self.artifacts_box.set_visible(button.get_active())
        if button.get_active():
            self._refresh_artifacts_list()

    def _refresh_artifacts_list(self):
        for child in list(self.artifacts_list.get_children()):
            self.artifacts_list.remove(child)
        if not self.session_id:
            return
        for artifact in memory.list_artifacts(self.session_id):
            row_label = Gtk.Label(label=f"[{artifact['kind']}] {artifact['label']}", xalign=0)
            row_label.set_margin_top(4)
            row_label.set_margin_bottom(4)
            row_label.set_margin_start(8)
            row_label.set_margin_end(8)
            row_label.set_ellipsize(Pango.EllipsizeMode.END)
            row = Gtk.ListBoxRow()
            row.add(row_label)
            row.artifact_data = artifact
            self.artifacts_list.add(row)
        self.artifacts_list.show_all()

    def _on_artifact_selected(self, _listbox, row):
        artifact = getattr(row, "artifact_data", None) if row else None
        self._current_artifact = artifact
        self.artifact_buffer.set_text(artifact["content"] if artifact else "")
        self.artifact_save_button.set_sensitive(bool(artifact))

    def _on_artifact_save_as(self, _button):
        artifact = self._current_artifact
        if not artifact:
            return
        dialog = Gtk.FileChooserDialog(
            title="Save artifact as", parent=self, action=Gtk.FileChooserAction.SAVE,
        )
        dialog.set_position(Gtk.WindowPosition.CENTER)
        dialog.set_do_overwrite_confirmation(True)
        dialog.set_current_name(os.path.basename(artifact["label"]) or "artifact.txt")
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_SAVE, Gtk.ResponseType.OK,
        )

        def on_response(dialog, response):
            path = dialog.get_filename() if response == Gtk.ResponseType.OK else None
            dialog.destroy()
            if path:
                with open(path, "w") as f:
                    f.write(artifact["content"])

        dialog.connect("response", on_response)
        dialog.show_all()

    def _on_new_thread(self, _button):
        if self.busy:
            return
        dialog = Gtk.FileChooserDialog(
            title="Choose a folder for the new thread",
            parent=self,
            action=Gtk.FileChooserAction.SELECT_FOLDER,
        )
        # Default position for a Gtk.Dialog with a parent is
        # CENTER_ON_PARENT, computed relative to the parent's own bounds
        # rather than clamped to the monitor — if the main window is
        # snapped to a screen edge, the "centered" dialog position can
        # extend past that same edge. CENTER instead centers on the
        # current monitor, staying fully on-screen regardless of where
        # the main window sits.
        dialog.set_position(Gtk.WindowPosition.CENTER)
        dialog.add_buttons(
            Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OPEN, Gtk.ResponseType.OK,
        )

        def on_response(dialog, response):
            path = dialog.get_filename() if response == Gtk.ResponseType.OK else None
            dialog.destroy()
            if path:
                self._switch_to(path)

        dialog.connect("response", on_response)
        dialog.show_all()

    # --- chat view rendering ---

    def _insert_tagged(self, segment, *tags):
        buf = self.buffer
        start = buf.get_end_iter().get_offset()
        buf.insert(buf.get_end_iter(), segment)
        end = buf.get_end_iter().get_offset()
        for tag in tags:
            buf.apply_tag(tag, buf.get_iter_at_offset(start), buf.get_iter_at_offset(end))

    def _insert_formatted(self, text, base_tag, image_map=None):
        """Insert text with **bold**, `code`, fenced code blocks, and
        ![alt](url) images rendered inline. Pango markup (used for a
        plain Label) isn't interpreted by a TextBuffer — tags/pixbufs
        have to be applied/inserted at explicit positions instead.

        image_map is {url: local_file_path} for images already
        downloaded by _download_reply_images — historical replay (no
        image_map) just shows the raw Markdown text instead of
        re-fetching on every thread switch."""
        image_map = image_map or {}
        parts = re.split(r"(```.*?```|!\[[^\]]*\]\([^)\s]+\))", text, flags=re.DOTALL)
        for part in parts:
            if part.startswith("```"):
                code = part.strip("`").strip()
                lines = code.split("\n")
                if lines and " " not in lines[0] and len(lines) > 1:
                    code = "\n".join(lines[1:])
                self._insert_tagged(code, base_tag, self.tag_code)
                continue

            if part.startswith("!["):
                match = IMAGE_MARKDOWN_RE.match(part)
                local_path = image_map.get(match.group(1)) if match else None
                if local_path:
                    try:
                        pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(local_path, 320, 320, True)
                        self.buffer.insert_pixbuf(self.buffer.get_end_iter(), pixbuf)
                        continue
                    except Exception:
                        pass
                self._insert_tagged(part, base_tag)  # fallback: show the raw Markdown text
                continue

            pos = 0
            for m in re.finditer(r"\*\*(.+?)\*\*|`([^`]+)`", part):
                if m.start() > pos:
                    self._insert_tagged(part[pos:m.start()], base_tag)
                if m.group(1) is not None:
                    self._insert_tagged(m.group(1), base_tag, self.tag_bold)
                else:
                    self._insert_tagged(m.group(2), base_tag, self.tag_code)
                pos = m.end()
            if pos < len(part):
                self._insert_tagged(part[pos:], base_tag)

    def _download_reply_images(self, text):
        """Download each ![alt](url) image referenced in a reply, once
        per URL, so _insert_formatted can embed the real picture instead
        of leaving Markdown syntax as literal text. Runs on the
        background agent thread (network I/O), not the GTK main thread.
        Local paths (e.g. from generate_image) are already on disk —
        those are just used directly, no download needed."""
        image_map = {}
        for url in IMAGE_MARKDOWN_RE.findall(text):
            if url in image_map:
                continue
            if os.path.isabs(url) and os.path.exists(url):
                image_map[url] = url
                continue
            try:
                os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
                ext = os.path.splitext(url.split("?")[0])[1]
                if not ext or len(ext) > 5:
                    ext = ".img"
                path = os.path.join(DOWNLOAD_DIR, f"{digest}{ext}")
                if not os.path.exists(path):
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    with open(path, "wb") as f:
                        f.write(resp.content)
                image_map[url] = path
            except Exception:
                pass  # left unresolved — _insert_formatted falls back to the raw Markdown text
        return image_map

    def _append_message(self, text, role, use_markup=False, image_map=None):
        tag = {"user": self.tag_user, "assistant": self.tag_assistant, "status": self.tag_status}[role]
        if self.buffer.get_char_count() > 0:
            self.buffer.insert(self.buffer.get_end_iter(), "\n\n")
        if use_markup:
            self._insert_formatted(text, tag, image_map)
        else:
            self._insert_tagged(text, tag)
        self._scroll_to_bottom()

    def _start_thinking(self, label="Liam is thinking"):
        """A transient, ticking status line — "Liam is thinking... (3s)"
        — for the gaps that otherwise show nothing but the small header
        spinner: waiting for the model's first response, and again
        between one tool call and the next (or the final answer). Always
        the last thing in the buffer while active. Needs *two* marks, not
        one: _anchor_mark sits before the "\n\n" separator (so
        _stop_thinking can wipe the separator plus the ticking text back
        to exactly how things looked before this call), while _text_mark
        sits right after that separator (so each tick only ever replaces
        the ticking text itself, never eating the separator too — that
        was the actual bug: a single mark before the separator meant the
        very first tick's delete-and-reinsert removed the separator and
        never put it back, cramming "Liam is thinking..." directly
        against the previous line with no blank line between them).
        Must be paired with _stop_thinking before any other text is
        appended."""
        if self._thinking_active:
            return
        self._thinking_active = True
        self._thinking_label = label
        self._thinking_started_at = time.time()
        self._thinking_anchor_mark = self.buffer.create_mark(None, self.buffer.get_end_iter(), True)
        if self.buffer.get_char_count() > 0:
            self.buffer.insert(self.buffer.get_end_iter(), "\n\n")
        self._thinking_text_mark = self.buffer.create_mark(None, self.buffer.get_end_iter(), True)
        self._render_thinking_tick()
        self._thinking_timer_id = GLib.timeout_add(500, self._render_thinking_tick)

    def _render_thinking_tick(self):
        if not self._thinking_active:
            return False
        start = self.buffer.get_iter_at_mark(self._thinking_text_mark)
        self.buffer.delete(start, self.buffer.get_end_iter())
        elapsed = int(time.time() - self._thinking_started_at)
        dots = "." * ((elapsed % 3) + 1)
        text = f"{self._thinking_label}{dots} ({elapsed}s)"
        self.buffer.insert_with_tags(
            self.buffer.get_iter_at_mark(self._thinking_text_mark), text, self.tag_status,
        )
        self._scroll_to_bottom()
        return True  # keep the timer repeating

    def _stop_thinking(self):
        if not self._thinking_active:
            return
        self._thinking_active = False
        GLib.source_remove(self._thinking_timer_id)
        self._thinking_timer_id = None
        start = self.buffer.get_iter_at_mark(self._thinking_anchor_mark)
        self.buffer.delete(start, self.buffer.get_end_iter())
        self.buffer.delete_mark(self._thinking_anchor_mark)
        self.buffer.delete_mark(self._thinking_text_mark)
        self._thinking_anchor_mark = None
        self._thinking_text_mark = None

    def _scroll_to_bottom(self):
        def do_scroll():
            adj = self.scroller.get_vadjustment()
            adj.set_value(adj.get_upper())
            return False
        GLib.idle_add(do_scroll)

    def _set_busy(self, busy):
        self.busy = busy
        self.entry.set_sensitive(not busy)
        self.send_button.set_sensitive(not busy)
        self.session_list.set_sensitive(not busy)
        self.external_sessions_toggle.set_sensitive(not busy)
        if not busy:
            # Re-enabling the entry doesn't give it back keyboard focus by
            # itself — without this, every reply leaves focus nowhere in
            # particular, and typing the next message needs a mouse click
            # on the entry first.
            self.entry.grab_focus()

    def _on_entry_paste_clipboard(self, entry):
        clipboard = Gtk.Clipboard.get_default(self.get_display())
        if not clipboard.wait_is_image_available():
            return  # let the default handler paste normally as text
        pixbuf = clipboard.wait_for_image()
        if pixbuf is not None:
            self._set_pending_image(pixbuf)
        # Gtk.Entry has no concept of an embedded image — without this,
        # the default paste handler still runs and inserts whatever text
        # representation (usually none/garbage) the clipboard also offers.
        entry.stop_emission_by_name("paste-clipboard")

    def _set_pending_image(self, pixbuf):
        os.makedirs(PASTE_DIR, exist_ok=True)
        path = os.path.join(PASTE_DIR, f"paste_{int(time.time() * 1000)}.png")
        pixbuf.savev(path, "png", [], [])
        self._pending_image_path = path
        thumb = pixbuf.scale_simple(48, 48, GdkPixbuf.InterpType.BILINEAR)
        self.pending_image_view.set_from_pixbuf(thumb)
        self.pending_image_box.set_visible(True)

    def _clear_pending_image(self):
        self._pending_image_path = None
        self.pending_image_box.set_visible(False)

    def _append_image(self, path):
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(path, 320, 320, True)
        except Exception:
            return
        if self.buffer.get_char_count() > 0:
            self.buffer.insert(self.buffer.get_end_iter(), "\n\n")
        self.buffer.insert_pixbuf(self.buffer.get_end_iter(), pixbuf)
        self._scroll_to_bottom()

    def _on_entry_key_press(self, _widget, event):
        if event.keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            if event.state & Gdk.ModifierType.SHIFT_MASK:
                return False  # let GTK insert the newline normally
            self._on_send(_widget)
            return True  # consume — don't also insert a newline
        return False

    def _on_send(self, _widget):
        if self.busy:
            return
        buf = self.entry.get_buffer()
        text = buf.get_text(*buf.get_bounds(), True).strip()
        image_path = self._pending_image_path
        if not text and not image_path:
            return
        buf.set_text("")
        display_text = text or "Read any text in this image."
        if image_path:
            self._append_image(image_path)
        self._append_message(display_text, "user")

        images_b64 = None
        if image_path:
            with open(image_path, "rb") as f:
                images_b64 = [base64.b64encode(f.read()).decode("ascii")]
        self._clear_pending_image()

        self._set_busy(True)
        self._start_thinking()
        threading.Thread(target=self._run_agent, args=(display_text, images_b64), daemon=True).start()

    def _idle_append(self, text, role, use_markup=False):
        """GLib.idle_add calls its target repeatedly forever unless it
        returns False/None. This wrapper exists solely to make sure that
        happens regardless of what _append_message itself returns."""
        self._append_message(text, role, use_markup)
        return False

    def _idle_append_between_thinking(self, text, role):
        """Tool-call and status lines land in the middle of a run, with
        the ticking "Liam is thinking..." line always at the buffer's
        tail — pause it, show the real line permanently, then resume
        thinking for whatever the model does next (another tool call, or
        the final answer)."""
        self._stop_thinking()
        self._append_message(text, role)
        self._start_thinking()
        return False

    def _on_tool_call(self, name, args):
        text = f"-> {name}({json.dumps(args)})"
        GLib.idle_add(self._idle_append_between_thinking, text, "status")

    def _on_status(self, text):
        GLib.idle_add(self._idle_append_between_thinking, text, "status")

    def _on_confirm(self, name, args):
        if name == "propose_lesson":
            return self._confirm_lesson(args)
        # The blocking here is deliberate and still needed — this runs on
        # the agent's background thread and must not return until the
        # user answers — but it's done via threading.Event(), not
        # Gtk.Dialog.run(). run() forces modal=True on the GTK side for
        # as long as it blocks, which is exactly what makes GNOME's
        # attach-modal-dialogs glue it to the main window; show()+
        # connect("response",...) gets the same "block the caller until
        # answered" behavior without ever setting modal at all.
        event = threading.Event()
        result = {"allow": False}

        def show_dialog():
            dialog = Gtk.MessageDialog(
                transient_for=self,
                message_type=Gtk.MessageType.QUESTION,
                buttons=Gtk.ButtonsType.NONE,
                text="Allow this action?",
                secondary_text=f"{name}({json.dumps(args)})",
            )
            dialog.add_buttons(
                "Deny", Gtk.ResponseType.NO,
                "Allow", Gtk.ResponseType.YES,
            )

            def on_response(dialog, response):
                result["allow"] = response == Gtk.ResponseType.YES
                dialog.destroy()
                event.set()

            dialog.connect("response", on_response)
            dialog.show_all()
            return False

        GLib.idle_add(show_dialog)
        event.wait()
        return result["allow"]

    def _confirm_lesson(self, args):
        """Special-cased confirm for propose_lesson: unlike the generic
        Allow/Deny dialog above, this lets the user actually rewrite
        Liam's proposed keywords/lesson before anything is saved.
        Mutating args in place is enough — _run_tool calls the tool with
        this same dict afterward, so an edit here reaches propose_lesson
        automatically, no new plumbing needed."""
        event = threading.Event()
        result = {"allow": False}

        def show_dialog():
            dialog = Gtk.Dialog(title="Save this lesson for next time?", transient_for=self)
            dialog.add_buttons(
                "Discard", Gtk.ResponseType.CANCEL,
                "Save", Gtk.ResponseType.OK,
            )
            box = dialog.get_content_area()
            box.set_spacing(8)
            box.set_margin_start(12)
            box.set_margin_end(12)
            box.set_margin_top(12)
            box.set_margin_bottom(12)

            box.add(Gtk.Label(label="Keywords (comma-separated):", xalign=0))
            keywords_entry = Gtk.Entry()
            keywords_entry.set_text(args.get("keywords", ""))
            box.add(keywords_entry)

            box.add(Gtk.Label(label="Lesson:", xalign=0))
            lesson_buffer = Gtk.TextBuffer()
            lesson_buffer.set_text(args.get("lesson", ""))
            lesson_view = Gtk.TextView(buffer=lesson_buffer)
            lesson_view.set_wrap_mode(Gtk.WrapMode.WORD)
            lesson_view.set_size_request(360, 100)
            box.add(lesson_view)

            def on_response(dialog, response):
                if response == Gtk.ResponseType.OK:
                    args["keywords"] = keywords_entry.get_text().strip() or args.get("keywords", "")
                    start, end = lesson_buffer.get_bounds()
                    lesson_text = lesson_buffer.get_text(start, end, True).strip()
                    args["lesson"] = lesson_text or args.get("lesson", "")
                    result["allow"] = True
                dialog.destroy()
                event.set()

            dialog.connect("response", on_response)
            dialog.show_all()
            return False

        GLib.idle_add(show_dialog)
        event.wait()
        return result["allow"]

    def _run_agent(self, text, images=None):
        agent = self.agent
        try:
            reply = agent.step(text, images=images)
        except Exception as exc:
            reply = f"[error] {exc}"
        image_map = self._download_reply_images(reply)
        GLib.idle_add(self._finish_reply, reply, image_map)

    def _finish_reply(self, reply, image_map=None):
        self._stop_thinking()
        self._append_message(reply, "assistant", use_markup=True, image_map=image_map)
        self._set_busy(False)
        return False


class LiamApp(Gtk.Application):
    def __init__(self, model, confirm):
        super().__init__(application_id=APP_ID)
        self.model = model
        self.confirm = confirm

    def do_activate(self):
        window = self.props.active_window
        if not window:
            window = LiamWindow(self, self.model, not self.confirm)
            # Patrick applies the placement once before the window is shown,
            # then repeats it after the first visible frame/map event.
            window.restore_saved_placement()
            window.show_all()
            # show_all() shows every child widget regardless of any
            # set_visible(False) called earlier in __init__ — it's not
            # "show everything that was already meant to be visible", it
            # unconditionally overrides that. The artifacts panel starts
            # closed; re-hide it now that show_all() has stomped on that.
            window.artifacts_box.set_visible(window.artifacts_toggle.get_active())
        window.present()

    def do_shutdown(self):
        window = self.props.active_window
        if isinstance(window, LiamWindow):
            window.flush_window_placement()
        Gtk.Application.do_shutdown(self)


def main():
    import argparse
    saved = liam_settings.load()
    parser = argparse.ArgumentParser(description="Liam - native GTK UI")
    parser.add_argument("--model", default=saved["model"] or DEFAULT_MODEL)
    parser.add_argument("--confirm", action="store_true", default=not saved["auto_confirm"])
    args = parser.parse_args()

    app = LiamApp(model=args.model, confirm=args.confirm)
    app.run([])


if __name__ == "__main__":
    main()
