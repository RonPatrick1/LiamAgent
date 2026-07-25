"""Scheduled prompts ("Routines") — DB storage plus the systemd --user
timer units that actually run them. A routine fires at its scheduled time
whether or not the Liam GUI is open, via `LiamAgent.py --routine <id>`
invoked headlessly by its own timer unit — an in-app timer would only ever
fire while someone happened to have the window open.
"""

import os
import subprocess

from .memory import _connect, _ensure_schema

SYSTEMD_USER_DIR = os.path.expanduser("~/.config/systemd/user")
LIAM_AGENT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LiamAgent.py",
)


def _row_to_dict(row):
    id_, session_id, prompt, schedule_kind, schedule_value, enabled, last_run_at = row
    return {
        "id": id_, "session_id": session_id, "prompt": prompt,
        "schedule_kind": schedule_kind, "schedule_value": schedule_value,
        "enabled": bool(enabled), "last_run_at": last_run_at,
    }


def create_routine(session_id, prompt, schedule_kind, schedule_value):
    """schedule_kind is 'daily' (schedule_value = 'HH:MM') or 'hourly'
    (schedule_value = the N in "every N hours"). Writes and enables the
    systemd unit immediately — a routine exists and runs, or doesn't
    exist at all; there's no disabled-but-half-created state."""
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO routines (session_id, prompt, schedule_kind, schedule_value) "
                "VALUES (%s, %s, %s, %s)",
                (session_id, prompt, schedule_kind, schedule_value),
            )
            routine_id = cur.lastrowid
    finally:
        conn.close()
    _write_units(routine_id, schedule_kind, schedule_value)
    return routine_id


def set_enabled(routine_id, enabled):
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE routines SET enabled = %s WHERE id = %s",
                (1 if enabled else 0, routine_id),
            )
    finally:
        conn.close()

    routine = get_routine(routine_id)
    if routine is None:
        return
    if enabled:
        _write_units(routine_id, routine["schedule_kind"], routine["schedule_value"])
    else:
        _remove_units(routine_id)


def delete_routine(routine_id):
    _remove_units(routine_id)
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM routines WHERE id = %s", (routine_id,))
    finally:
        conn.close()


def get_routine(routine_id):
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, session_id, prompt, schedule_kind, schedule_value, enabled, last_run_at "
                "FROM routines WHERE id = %s",
                (routine_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return _row_to_dict(row) if row else None


def list_routines():
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, session_id, prompt, schedule_kind, schedule_value, enabled, last_run_at "
                "FROM routines ORDER BY id DESC"
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    return [_row_to_dict(row) for row in rows]


def mark_ran(routine_id):
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE routines SET last_run_at = CURRENT_TIMESTAMP WHERE id = %s",
                (routine_id,),
            )
    finally:
        conn.close()


def _service_path(routine_id):
    return os.path.join(SYSTEMD_USER_DIR, f"liam-routine-{routine_id}.service")


def _timer_path(routine_id):
    return os.path.join(SYSTEMD_USER_DIR, f"liam-routine-{routine_id}.timer")


def _on_calendar(schedule_kind, schedule_value):
    if schedule_kind == "daily":
        hh, mm = schedule_value.split(":")
        return f"*-*-* {int(hh):02d}:{int(mm):02d}:00"
    if schedule_kind == "hourly":
        return f"*-*-* 0/{int(schedule_value)}:00:00"
    raise ValueError(f"unknown schedule_kind: {schedule_kind}")


def _write_units(routine_id, schedule_kind, schedule_value):
    os.makedirs(SYSTEMD_USER_DIR, exist_ok=True)
    on_calendar = _on_calendar(schedule_kind, schedule_value)

    service = (
        "[Unit]\n"
        f"Description=Liam routine {routine_id}\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"ExecStart=/usr/bin/python3 {LIAM_AGENT_PATH} --routine {routine_id}\n"
    )
    timer = (
        "[Unit]\n"
        f"Description=Liam routine {routine_id} schedule\n\n"
        "[Timer]\n"
        f"OnCalendar={on_calendar}\n"
        "Persistent=true\n\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )
    with open(_service_path(routine_id), "w") as f:
        f.write(service)
    with open(_timer_path(routine_id), "w") as f:
        f.write(timer)

    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", f"liam-routine-{routine_id}.timer"],
        capture_output=True,
    )


def _remove_units(routine_id):
    subprocess.run(
        ["systemctl", "--user", "disable", "--now", f"liam-routine-{routine_id}.timer"],
        capture_output=True,
    )
    for path in (_service_path(routine_id), _timer_path(routine_id)):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
