"""Persistent conversation memory backed by MySQL.

The database location is fixed on purpose: it's infrastructure, not
something the agent should rediscover or guess at each run. Only the
credentials come from the environment (.env).
"""

import os
import re

import pymysql

_REPETITION_RE = re.compile(r"(.{2,30}?)\1{6,}", re.DOTALL)


def _looks_corrupted(content):
    """Catches degenerate-generation output (a known local-model failure
    mode: a short token/phrase repeated dozens of times) so it never gets
    persisted and doesn't poison future context on reload."""
    return bool(_REPETITION_RE.search(content))

DB_HOST = "192.168.0.136"
DB_PORT = 3306
DB_NAME = "liams_memory"

DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")


def _connect():
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        autocommit=True,
        connect_timeout=5,
    )


def _ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                role VARCHAR(16) NOT NULL,
                content MEDIUMTEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INT AUTO_INCREMENT PRIMARY KEY,
                content MEDIUMTEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS search_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                query TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_message(role, content):
    """Append one message to persistent history. Failures are non-fatal —
    memory is a nice-to-have, not something that should crash the agent."""
    if _looks_corrupted(content):
        print("[memory] refusing to save degenerate/corrupted output")
        return
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages (role, content) VALUES (%s, %s)",
                    (role, content),
                )
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to save message: {exc}")


def load_recent_messages(limit=20):
    """Return the last `limit` messages, oldest first, as a list of
    {"role": ..., "content": ...} dicts. Returns [] on any failure."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content FROM messages ORDER BY id DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [{"role": role, "content": content} for role, content in reversed(rows)]
    except Exception as exc:
        print(f"[memory] failed to load history: {exc}")
        return []


def query_memory(limit=20, keyword=None):
    """Read-only introspection of the messages table, for the agent (or
    the user) to inspect its own memory directly. Returns a formatted
    string, not raw rows."""
    limit = min(max(int(limit), 1), 50)
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                if keyword:
                    cur.execute(
                        "SELECT id, role, content, created_at FROM messages "
                        "WHERE content LIKE %s ORDER BY id DESC LIMIT %s",
                        (f"%{keyword}%", limit),
                    )
                else:
                    cur.execute(
                        "SELECT id, role, content, created_at FROM messages "
                        "ORDER BY id DESC LIMIT %s",
                        (limit,),
                    )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as exc:
        return f"query_memory failed: {exc}"

    if not rows:
        return "No matching messages found."
    lines = [f"[{DB_HOST}:{DB_PORT}/{DB_NAME}.messages]"]
    for id_, role, content, created_at in reversed(rows):
        lines.append(f"#{id_} ({created_at}) {role}: {content}")
    return "\n".join(lines)


def remember(content):
    """Save something the user explicitly wants remembered — a fact,
    preference, or reminder — separate from the raw conversation log.
    Skips inserting if an identical note already exists, so retries or
    repeated confirmations don't create duplicates."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM notes WHERE content = %s LIMIT 1", (content,))
                existing = cur.fetchone()
                if existing:
                    return f"Already remembered as #{existing[0]} — not duplicated."
                cur.execute("INSERT INTO notes (content) VALUES (%s)", (content,))
                new_id = cur.lastrowid
        finally:
            conn.close()
        return f"Remembered as #{new_id}."
    except Exception as exc:
        return f"Failed to save note: {exc}"


def forget(note_id=None, keyword=None):
    """Delete note(s) by exact id or by a content keyword match."""
    if not note_id and not keyword:
        return "Specify either note_id or keyword to identify what to forget."
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                if note_id:
                    cur.execute("DELETE FROM notes WHERE id = %s", (note_id,))
                else:
                    cur.execute("DELETE FROM notes WHERE content LIKE %s", (f"%{keyword}%",))
                deleted = cur.rowcount
        finally:
            conn.close()
        if deleted == 0:
            return "No matching notes found to delete."
        return f"Deleted {deleted} note(s)."
    except Exception as exc:
        return f"Failed to delete note(s): {exc}"


def load_recent_notes(limit=50):
    """Return recent notes, oldest first, as plain strings."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM notes ORDER BY id DESC LIMIT %s",
                    (limit,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [content for (content,) in reversed(rows)]
    except Exception as exc:
        print(f"[memory] failed to load notes: {exc}")
        return []


def recall_notes(keyword=None, limit=50):
    """Read-only introspection of saved notes, for the agent to search
    through what the user has explicitly asked it to remember."""
    limit = min(max(int(limit), 1), 200)
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                if keyword:
                    cur.execute(
                        "SELECT id, content, created_at FROM notes "
                        "WHERE content LIKE %s ORDER BY id DESC LIMIT %s",
                        (f"%{keyword}%", limit),
                    )
                else:
                    cur.execute(
                        "SELECT id, content, created_at FROM notes "
                        "ORDER BY id DESC LIMIT %s",
                        (limit,),
                    )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as exc:
        return f"recall_notes failed: {exc}"

    if not rows:
        return "No matching notes found."
    lines = [f"[{DB_HOST}:{DB_PORT}/{DB_NAME}.notes]"]
    for id_, content, created_at in reversed(rows):
        lines.append(f"#{id_} ({created_at}): {content}")
    return "\n".join(lines)


# Brave's free tier is $5/month credit at $5 per 1,000 requests — i.e. 1,000
# free searches/month before anything is billable.
BRAVE_FREE_REQUESTS_PER_MONTH = 1000
BRAVE_PRICE_PER_1000 = 5.0


def log_search(query):
    """Record one real Brave Search API call, so usage can be tracked
    locally instead of checking Brave's dashboard."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute("INSERT INTO search_log (query) VALUES (%s)", (query,))
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to log search: {exc}")


def get_search_usage():
    """Summarize Brave Search usage this month from our own local log."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM search_log "
                    "WHERE created_at >= DATE_FORMAT(NOW(), '%Y-%m-01')"
                )
                (month_count,) = cur.fetchone()
                cur.execute(
                    "SELECT COUNT(*) FROM search_log WHERE DATE(created_at) = CURDATE()"
                )
                (today_count,) = cur.fetchone()
        finally:
            conn.close()
    except Exception as exc:
        return f"Failed to read search usage: {exc}"

    free_remaining = max(0, BRAVE_FREE_REQUESTS_PER_MONTH - month_count)
    billable = max(0, month_count - BRAVE_FREE_REQUESTS_PER_MONTH)
    cost = billable / 1000 * BRAVE_PRICE_PER_1000
    summary = (
        f"Brave Search usage this month (tracked locally, since Liam started "
        f"logging this): {month_count} requests ({today_count} today). "
        f"{free_remaining} of the {BRAVE_FREE_REQUESTS_PER_MONTH} free "
        f"monthly requests remain."
    )
    if billable:
        summary += f" ~{billable} over the free tier, roughly ${cost:.2f} billable."
    return summary
