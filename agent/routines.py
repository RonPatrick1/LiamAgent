"""Scheduled prompts ("Routines") — DB storage plus the systemd --user
timer units that actually run them. A routine fires at its scheduled time
whether or not the Liam GUI is open, via `LiamAgent.py --routine <id>`
invoked headlessly by its own timer unit — an in-app timer would only ever
fire while someone happened to have the window open.
"""

import os
import subprocess
from datetime import datetime

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
    """Create a once, daily, or hourly routine and verify its timer.

    A routine exists and runs, or doesn't exist at all; a systemd failure
    rolls the database row and unit files back instead of leaving an
    enabled-but-fictional schedule behind.
    """
    # Validate before inserting. A malformed calendar must not leave a DB
    # row that looks enabled even though no timer could have been created.
    _timer_schedule_lines(schedule_kind, schedule_value)
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
    try:
        _write_units(routine_id, schedule_kind, schedule_value)
    except Exception:
        _remove_units(routine_id)
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM routines WHERE id = %s", (routine_id,))
        finally:
            conn.close()
        raise
    return routine_id


def set_enabled(routine_id, enabled):
    routine = get_routine(routine_id)
    if routine is None:
        return
    if enabled:
        # Do this first so the database cannot claim an enabled schedule
        # when systemd rejected or failed to start its timer.
        _write_units(routine_id, routine["schedule_kind"], routine["schedule_value"])
    else:
        _remove_units(routine_id)

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


def enqueue_matrix_delivery(routine_id, room_id, content):
    """Queue a routine result for the encrypted Matrix bot to deliver."""
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO routine_deliveries (routine_id, room_id, content) "
                "VALUES (%s, %s, %s)",
                (routine_id, room_id, content),
            )
            return cur.lastrowid
    finally:
        conn.close()


def claim_matrix_delivery():
    """Atomically claim one pending (or abandoned) outbound result."""
    conn = _connect()
    try:
        conn.begin()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, routine_id, room_id, content, attempts "
                "FROM routine_deliveries "
                "WHERE (status = 'pending' OR "
                "      (status = 'delivering' AND claimed_at < CURRENT_TIMESTAMP - INTERVAL 5 MINUTE)) "
                "AND attempts < 5 ORDER BY id LIMIT 1 FOR UPDATE"
            )
            row = cur.fetchone()
            if row is None:
                conn.commit()
                return None
            delivery_id, routine_id, room_id, content, attempts = row
            cur.execute(
                "UPDATE routine_deliveries SET status = 'delivering', "
                "attempts = attempts + 1, claimed_at = CURRENT_TIMESTAMP, last_error = NULL "
                "WHERE id = %s",
                (delivery_id,),
            )
        conn.commit()
        return {
            "id": delivery_id,
            "routine_id": routine_id,
            "room_id": room_id,
            "content": content,
            "attempts": attempts + 1,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def resolve_matrix_delivery(delivery_id, delivered, error=None):
    conn = _connect()
    try:
        with conn.cursor() as cur:
            if delivered:
                cur.execute(
                    "UPDATE routine_deliveries SET status = 'delivered', "
                    "delivered_at = CURRENT_TIMESTAMP, last_error = NULL WHERE id = %s",
                    (delivery_id,),
                )
            else:
                cur.execute(
                    "UPDATE routine_deliveries SET "
                    "status = IF(attempts >= 5, 'failed', 'pending'), last_error = %s "
                    "WHERE id = %s",
                    ((error or "delivery failed")[:2000], delivery_id),
                )
            return cur.rowcount > 0
    finally:
        conn.close()


def _service_path(routine_id):
    return os.path.join(SYSTEMD_USER_DIR, f"liam-routine-{routine_id}.service")


def _timer_path(routine_id):
    return os.path.join(SYSTEMD_USER_DIR, f"liam-routine-{routine_id}.timer")


def _on_calendar(schedule_kind, schedule_value):
    if schedule_kind == "once":
        run_at = datetime.strptime(schedule_value, "%Y-%m-%d %H:%M:%S")
        if run_at <= datetime.now():
            raise ValueError("one-time routine must be scheduled in the future")
        return run_at.strftime("%Y-%m-%d %H:%M:%S")
    if schedule_kind == "daily":
        parsed = datetime.strptime(schedule_value, "%H:%M")
        return f"*-*-* {parsed.hour:02d}:{parsed.minute:02d}:00"
    if schedule_kind == "hourly":
        hours = int(schedule_value)
        if not 1 <= hours <= 168:
            raise ValueError("hourly interval must be between 1 and 168 hours")
        return f"*-*-* 0/{hours}:00:00"
    raise ValueError(f"unknown schedule_kind: {schedule_kind}")


def _timer_schedule_lines(schedule_kind, schedule_value):
    """Validate a schedule and return its systemd [Timer] directives."""
    if schedule_kind == "minutely":
        try:
            minutes = int(schedule_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("minute interval must be a whole number") from exc
        if not 1 <= minutes <= 1440:
            raise ValueError("minute interval must be between 1 and 1440 minutes")
        # A monotonic interval supports values such as 90 minutes without
        # forcing them into a wall-clock calendar expression. OnActiveSec
        # handles the first run; OnUnitActiveSec repeats after each run.
        return f"OnActiveSec={minutes}min\nOnUnitActiveSec={minutes}min"
    return f"OnCalendar={_on_calendar(schedule_kind, schedule_value)}\nPersistent=true"


def _write_units(routine_id, schedule_kind, schedule_value):
    os.makedirs(SYSTEMD_USER_DIR, exist_ok=True)
    schedule_lines = _timer_schedule_lines(schedule_kind, schedule_value)

    service = (
        "[Unit]\n"
        f"Description=Liam routine {routine_id}\n\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"WorkingDirectory={os.path.dirname(LIAM_AGENT_PATH)}\n"
        f"ExecStart=/usr/bin/python3 {LIAM_AGENT_PATH} --routine {routine_id}\n"
    )
    timer = (
        "[Unit]\n"
        f"Description=Liam routine {routine_id} schedule\n\n"
        "[Timer]\n"
        f"{schedule_lines}\n"
        "AccuracySec=1s\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )
    with open(_service_path(routine_id), "w") as f:
        f.write(service)
    with open(_timer_path(routine_id), "w") as f:
        f.write(timer)

    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", f"liam-routine-{routine_id}.timer"],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["systemctl", "--user", "is-active", f"liam-routine-{routine_id}.timer"],
        capture_output=True, text=True, check=True,
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
