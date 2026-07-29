"""Persistent conversation memory backed by MySQL.

The database location is fixed on purpose: it's infrastructure, not
something the agent should rediscover or guess at each run. Only the
credentials come from the environment (.env).
"""

import hashlib
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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                folder_path VARCHAR(1024) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS session_folders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT NOT NULL,
                folder_path VARCHAR(1024) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY session_folder_unique (session_id, folder_path)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS groups (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT NOT NULL,
                kind VARCHAR(16) NOT NULL,
                label VARCHAR(1024) NOT NULL,
                content MEDIUMTEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS routines (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT NOT NULL,
                prompt MEDIUMTEXT NOT NULL,
                schedule_kind VARCHAR(16) NOT NULL,
                schedule_value VARCHAR(32) NOT NULL,
                enabled TINYINT(1) NOT NULL DEFAULT 1,
                last_run_at TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS routine_deliveries (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                routine_id INT NOT NULL,
                room_id VARCHAR(512) NOT NULL,
                content MEDIUMTEXT NOT NULL,
                status VARCHAR(16) NOT NULL DEFAULT 'pending',
                attempts INT NOT NULL DEFAULT 0,
                last_error TEXT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                claimed_at TIMESTAMP NULL,
                delivered_at TIMESTAMP NULL,
                INDEX routine_delivery_status_created (status, created_at),
                INDEX routine_delivery_routine (routine_id)
            )
            """
        )
        # Renamed from "gotchas" -- same table, just a less silly name. Only
        # rename if a fresh "lessons" table doesn't already exist (upgrade
        # path); a brand-new install just gets "lessons" directly below.
        cur.execute("SHOW TABLES LIKE 'gotchas'")
        if cur.fetchone():
            cur.execute("SHOW TABLES LIKE 'lessons'")
            if not cur.fetchone():
                cur.execute("RENAME TABLE gotchas TO lessons")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lessons (
                id INT AUTO_INCREMENT PRIMARY KEY,
                keywords VARCHAR(512) NOT NULL,
                lesson MEDIUMTEXT NOT NULL,
                source_session_id INT NULL,
                hit_count INT NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Kept only as an upgrade source for installations that used the
        # first auto-learning prototype. New observations go into lessons +
        # lesson_events below, with a shared lifecycle and provenance model.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS auto_lessons (
                id INT AUTO_INCREMENT PRIMARY KEY,
                keywords VARCHAR(512) NOT NULL,
                lesson MEDIUMTEXT NOT NULL,
                detector VARCHAR(64) NOT NULL,
                source_session_id INT NULL,
                hit_count INT NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Lessons now share one lifecycle regardless of where they came
        # from.  Keep ALTERs idempotent because _ensure_schema runs from
        # every frontend and existing installations may be at any older
        # schema revision.
        lesson_columns = {
            "status": "VARCHAR(16) NOT NULL DEFAULT 'active'",
            "origin": "VARCHAR(32) NOT NULL DEFAULT 'legacy'",
            "scope_kind": "VARCHAR(16) NOT NULL DEFAULT 'global'",
            "scope_value": "VARCHAR(1024) NULL",
            "detector": "VARCHAR(64) NULL",
            "fingerprint": "CHAR(64) NULL",
            "occurrence_count": "INT NOT NULL DEFAULT 1",
            "success_count": "INT NOT NULL DEFAULT 0",
            "failure_count": "INT NOT NULL DEFAULT 0",
            "consecutive_failure_count": "INT NOT NULL DEFAULT 0",
            "last_seen_at": "TIMESTAMP NULL",
            "last_used_at": "TIMESTAMP NULL",
            "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
            "merged_into_id": "INT NULL",
        }
        for column, definition in lesson_columns.items():
            cur.execute(f"SHOW COLUMNS FROM lessons LIKE '{column}'")
            if not cur.fetchone():
                cur.execute(f"ALTER TABLE lessons ADD COLUMN {column} {definition}")
        cur.execute("SHOW INDEX FROM lessons WHERE Key_name = 'lesson_fingerprint_unique'")
        if not cur.fetchone():
            cur.execute(
                "ALTER TABLE lessons ADD UNIQUE KEY lesson_fingerprint_unique (fingerprint)"
            )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lesson_events (
                id INT AUTO_INCREMENT PRIMARY KEY,
                lesson_id INT NOT NULL,
                event_kind VARCHAR(32) NOT NULL,
                detector VARCHAR(64) NULL,
                source_session_id INT NULL,
                source_channel VARCHAR(32) NULL,
                source_actor VARCHAR(255) NULL,
                evidence MEDIUMTEXT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX lesson_events_lesson (lesson_id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lesson_uses (
                id INT AUTO_INCREMENT PRIMARY KEY,
                lesson_id INT NOT NULL,
                session_id INT NULL,
                channel VARCHAR(32) NULL,
                outcome VARCHAR(16) NOT NULL DEFAULT 'unknown',
                detector VARCHAR(64) NULL,
                failure_signature VARCHAR(255) NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP NULL,
                INDEX lesson_uses_lesson (lesson_id)
            )
            """
        )
        # Older auto_lessons were synthesized from a failure alone, before
        # Liam required an observed successful recovery. Preserve them for
        # review, but never silently promote that legacy evidence to active.
        cur.execute(
            """
            INSERT INTO lessons (
                keywords, lesson, source_session_id, hit_count, created_at,
                status, origin, scope_kind, detector, fingerprint,
                scope_value, occurrence_count, last_seen_at
            )
            SELECT
                a.keywords, a.lesson, a.source_session_id, a.hit_count,
                a.created_at, 'pending', 'legacy_auto', 'tool', a.detector,
                SHA2(CONCAT('legacy-auto:', a.id), 256),
                'run_shell_command', 1, a.created_at
            FROM auto_lessons a
            LEFT JOIN lessons l
              ON l.fingerprint = SHA2(CONCAT('legacy-auto:', a.id), 256)
            WHERE l.id IS NULL
            """
        )
        cur.execute("SHOW COLUMNS FROM messages LIKE 'session_id'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE messages ADD COLUMN session_id INT NULL")
            legacy_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cur.execute(
                "INSERT INTO sessions (title, folder_path) VALUES (%s, %s)",
                (os.path.basename(legacy_folder) or legacy_folder, legacy_folder),
            )
            legacy_id = cur.lastrowid
            cur.execute(
                "UPDATE messages SET session_id = %s WHERE session_id IS NULL",
                (legacy_id,),
            )
        cur.execute("SHOW COLUMNS FROM sessions LIKE 'pinned'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE sessions ADD COLUMN pinned TINYINT(1) NOT NULL DEFAULT 0")
        cur.execute("SHOW COLUMNS FROM sessions LIKE 'unread'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE sessions ADD COLUMN unread TINYINT(1) NOT NULL DEFAULT 0")
        cur.execute("SHOW COLUMNS FROM sessions LIKE 'archived'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE sessions ADD COLUMN archived TINYINT(1) NOT NULL DEFAULT 0")
        cur.execute("SHOW COLUMNS FROM sessions LIKE 'group_id'")
        if not cur.fetchone():
            cur.execute("ALTER TABLE sessions ADD COLUMN group_id INT NULL")
        # Forking needs more than one thread able to share a folder_path —
        # drop the uniqueness this table was originally created with.
        cur.execute("SHOW INDEX FROM sessions WHERE Key_name = 'folder_path_unique'")
        if cur.fetchone():
            cur.execute("ALTER TABLE sessions DROP INDEX folder_path_unique")
        cur.execute("SHOW COLUMNS FROM notes LIKE 'session_id'")
        if not cur.fetchone():
            # NULL stays the existing global/legacy notes pool (unchanged
            # behavior for the GUI/CLI, and for whichever bucket is
            # designated to share it — see agent/server.py). A real value
            # here isolates a bucket's notes from every other bucket's.
            cur.execute("ALTER TABLE notes ADD COLUMN session_id INT NULL")


def save_message(role, content, session_id=None):
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
                    "INSERT INTO messages (role, content, session_id) VALUES (%s, %s, %s)",
                    (role, content, session_id),
                )
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to save message: {exc}")


def load_recent_messages(limit=20, session_id=None):
    """Return the last `limit` messages, oldest first, as a list of
    {"role": ..., "content": ...} dicts. Returns [] on any failure."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                if session_id is not None:
                    cur.execute(
                        "SELECT role, content FROM messages WHERE session_id = %s "
                        "ORDER BY id DESC LIMIT %s",
                        (session_id, limit),
                    )
                else:
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


def query_memory(limit=20, keyword=None, session_id=None):
    """Read-only introspection of the messages table, for the agent (or
    the user) to inspect its own memory directly. Returns a formatted
    string, not raw rows."""
    limit = min(max(int(limit), 1), 50)
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                clauses = []
                params = []
                if session_id is not None:
                    clauses.append("session_id = %s")
                    params.append(session_id)
                if keyword:
                    clauses.append("content LIKE %s")
                    params.append(f"%{keyword}%")
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                params.append(limit)
                cur.execute(
                    f"SELECT id, role, content, created_at FROM messages "
                    f"{where} ORDER BY id DESC LIMIT %s",
                    params,
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


def get_or_create_session(folder_path, title=None):
    """Return the session id for a given folder, creating one (and
    touching its last_active_at) if it doesn't exist yet. Reopening a
    folder rejoins its oldest thread rather than forking a duplicate —
    forking is now possible (fork_session), but isn't what a plain
    "+"/CLI-launch-in-this-folder should do."""
    folder_path = os.path.abspath(os.path.expanduser(folder_path))
    title = title or os.path.basename(folder_path.rstrip("/")) or folder_path
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM sessions WHERE folder_path = %s ORDER BY id ASC LIMIT 1",
                (folder_path,),
            )
            row = cur.fetchone()
            if row:
                session_id = row[0]
                cur.execute(
                    "UPDATE sessions SET last_active_at = CURRENT_TIMESTAMP WHERE id = %s",
                    (session_id,),
                )
                return session_id
            cur.execute(
                "INSERT INTO sessions (title, folder_path) VALUES (%s, %s)",
                (title, folder_path),
            )
            return cur.lastrowid
    finally:
        conn.close()


def get_session(session_id):
    """Return one session as a dict (same shape as list_sessions' items,
    minus group info), or None if it doesn't exist — e.g. for a routine
    resolving which thread it runs against."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, title, folder_path, pinned, unread, archived "
                    "FROM sessions WHERE id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        if not row:
            return None
        id_, title, folder_path, pinned, unread, archived = row
        return {
            "id": id_, "title": title, "folder_path": folder_path,
            "pinned": bool(pinned), "unread": bool(unread), "archived": bool(archived),
        }
    except Exception as exc:
        print(f"[memory] failed to get session: {exc}")
        return None


def list_sessions(include_archived=False):
    """Return sessions as a list of dicts, pinned first, then grouped
    (named groups before ungrouped), then most-recently-active within each
    bucket — matching Claude.ai's own pinned-chats-stay-on-top behavior,
    plus this app's own grouping. Archived threads are left out unless
    include_archived=True, same as an archived chat not cluttering the
    normal list."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                where = "" if include_archived else "WHERE s.archived = 0"
                cur.execute(
                    f"SELECT s.id, s.title, s.folder_path, s.last_active_at, s.pinned, "
                    f"s.unread, s.archived, s.group_id, g.name "
                    f"FROM sessions s LEFT JOIN groups g ON s.group_id = g.id "
                    f"{where} "
                    f"ORDER BY s.pinned DESC, g.name IS NULL, g.name ASC, s.last_active_at DESC"
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {
                "id": id_, "title": title, "folder_path": folder_path,
                "last_active_at": last_active_at,
                "pinned": bool(pinned), "unread": bool(unread), "archived": bool(archived),
                "group_id": group_id, "group_name": group_name,
            }
            for id_, title, folder_path, last_active_at, pinned, unread, archived, group_id, group_name in rows
        ]
    except Exception as exc:
        print(f"[memory] failed to list sessions: {exc}")
        return []


def list_groups():
    """Return all groups as a list of {"id", "name"} dicts, alphabetical."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT id, name FROM groups ORDER BY name ASC")
                rows = cur.fetchall()
        finally:
            conn.close()
        return [{"id": id_, "name": name} for id_, name in rows]
    except Exception as exc:
        print(f"[memory] failed to list groups: {exc}")
        return []


def create_group(name):
    """Create a new group and return its id."""
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO groups (name) VALUES (%s)", (name,))
            return cur.lastrowid
    finally:
        conn.close()


def set_session_group(session_id, group_id):
    """Assign (or clear, with group_id=None) which group a thread belongs
    to."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET group_id = %s WHERE id = %s",
                    (group_id, session_id),
                )
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to set session group: {exc}")


def fork_session(session_id):
    """Duplicate a thread into a new, independent one pointed at the same
    folder — a real copy of its message history, not a reference to the
    original. Returns the new session's id, or None on failure."""
    conn = _connect()
    try:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT title, folder_path, group_id FROM sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            title, folder_path, group_id = row
            cur.execute(
                "INSERT INTO sessions (title, folder_path, group_id) VALUES (%s, %s, %s)",
                (f"{title} (fork)", folder_path, group_id),
            )
            new_id = cur.lastrowid
            cur.execute(
                "INSERT INTO messages (role, content, session_id) "
                "SELECT role, content, %s FROM messages WHERE session_id = %s ORDER BY id",
                (new_id, session_id),
            )
            cur.execute(
                "INSERT INTO session_folders (session_id, folder_path) "
                "SELECT %s, folder_path FROM session_folders WHERE session_id = %s",
                (new_id, session_id),
            )
            return new_id
    except Exception as exc:
        print(f"[memory] failed to fork session: {exc}")
        return None
    finally:
        conn.close()


def add_session_folder(session_id, folder_path):
    """Attach an additional folder to a thread — read/write access for it
    is scoped by absolute path, not as the thread's default cwd (that
    stays sessions.folder_path). Silently ignores a duplicate."""
    folder_path = os.path.abspath(os.path.expanduser(folder_path))
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT IGNORE INTO session_folders (session_id, folder_path) VALUES (%s, %s)",
                    (session_id, folder_path),
                )
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to add session folder: {exc}")


def remove_session_folder(folder_id):
    """Detach an additional folder from a thread by its session_folders
    row id (not the primary folder — that's part of the session itself)."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM session_folders WHERE id = %s", (folder_id,))
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to remove session folder: {exc}")


def list_session_folders(session_id):
    """Return this thread's additional folders (not the primary one) as
    a list of {"id", "folder_path"} dicts."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, folder_path FROM session_folders WHERE session_id = %s ORDER BY id",
                    (session_id,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [{"id": id_, "folder_path": folder_path} for id_, folder_path in rows]
    except Exception as exc:
        print(f"[memory] failed to list session folders: {exc}")
        return []


def touch_session(session_id):
    """Bump a session's last_active_at, e.g. when it becomes the active
    thread again after switching away and back."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET last_active_at = CURRENT_TIMESTAMP WHERE id = %s",
                    (session_id,),
                )
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to touch session: {exc}")


def rename_session(session_id, title):
    """Rename a thread's display title — e.g. from the GUI sidebar's
    rename action. Only the title changes; folder_path (the session's
    real identity) is untouched."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET title = %s WHERE id = %s",
                    (title, session_id),
                )
        finally:
            conn.close()
        return f"Renamed to '{title}'."
    except Exception as exc:
        return f"Failed to rename session: {exc}"


def _set_bool_column(session_id, column, value):
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE sessions SET {column} = %s WHERE id = %s",
                    (1 if value else 0, session_id),
                )
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to set {column}: {exc}")


def set_pinned(session_id, pinned):
    """Pin or unpin a thread — pinned threads sort above the rest in
    list_sessions(), matching Claude.ai's own pin behavior."""
    _set_bool_column(session_id, "pinned", pinned)


def set_unread(session_id, unread):
    """Mark a thread read/unread — a plain visual flag, no notification
    semantics attached."""
    _set_bool_column(session_id, "unread", unread)


def set_archived(session_id, archived):
    """Archive or unarchive a thread — archived threads are hidden from
    list_sessions() by default but not deleted (see delete_session for
    that)."""
    _set_bool_column(session_id, "archived", archived)


def delete_session(session_id):
    """Delete a thread and its message history — e.g. from the GUI
    sidebar's delete action. Removes the messages tied to this session
    first, then the session row itself, so nothing is left orphaned."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
                cur.execute("DELETE FROM session_folders WHERE session_id = %s", (session_id,))
                cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
        finally:
            conn.close()
        return "Deleted."
    except Exception as exc:
        return f"Failed to delete session: {exc}"


def remember(content, session_id=None):
    """Save something the user explicitly wants remembered — a fact,
    preference, or reminder — separate from the raw conversation log.
    Skips inserting if an identical note already exists in the same
    bucket, so retries or repeated confirmations don't create duplicates.

    session_id=None is the existing global notes pool (unchanged GUI/CLI
    behavior, and whichever messenger bucket is designated to share it);
    a real value isolates a bucket's notes from every other bucket's —
    the NULL-safe <=> comparison is what makes "same bucket" match
    correctly whether that bucket is NULL or a real id."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM notes WHERE content = %s AND session_id <=> %s LIMIT 1",
                    (content, session_id),
                )
                existing = cur.fetchone()
                if existing:
                    return f"Already remembered as #{existing[0]} — not duplicated."
                cur.execute(
                    "INSERT INTO notes (content, session_id) VALUES (%s, %s)",
                    (content, session_id),
                )
                new_id = cur.lastrowid
        finally:
            conn.close()
        return f"Remembered as #{new_id}."
    except Exception as exc:
        return f"Failed to save note: {exc}"


def forget(note_id=None, keyword=None, session_id=None):
    """Delete note(s) by exact id or by a content keyword match — scoped
    to the same bucket (session_id) the note was remembered in, so one
    bucket can't reach into another's notes even by guessing an id."""
    if not note_id and not keyword:
        return "Specify either note_id or keyword to identify what to forget."
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                if note_id:
                    cur.execute(
                        "DELETE FROM notes WHERE id = %s AND session_id <=> %s",
                        (note_id, session_id),
                    )
                else:
                    cur.execute(
                        "DELETE FROM notes WHERE content LIKE %s AND session_id <=> %s",
                        (f"%{keyword}%", session_id),
                    )
                deleted = cur.rowcount
        finally:
            conn.close()
        if deleted == 0:
            return "No matching notes found to delete."
        return f"Deleted {deleted} note(s)."
    except Exception as exc:
        return f"Failed to delete note(s): {exc}"


def load_recent_notes(limit=50, session_id=None):
    """Return recent notes for this bucket, oldest first, as plain
    strings."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM notes WHERE session_id <=> %s ORDER BY id DESC LIMIT %s",
                    (session_id, limit),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [content for (content,) in reversed(rows)]
    except Exception as exc:
        print(f"[memory] failed to load notes: {exc}")
        return []


def recall_notes(keyword=None, limit=50, session_id=None):
    """Read-only introspection of this bucket's saved notes, for the
    agent to search through what the user has explicitly asked it to
    remember."""
    limit = min(max(int(limit), 1), 200)
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                if keyword:
                    cur.execute(
                        "SELECT id, content, created_at FROM notes "
                        "WHERE content LIKE %s AND session_id <=> %s ORDER BY id DESC LIMIT %s",
                        (f"%{keyword}%", session_id, limit),
                    )
                else:
                    cur.execute(
                        "SELECT id, content, created_at FROM notes "
                        "WHERE session_id <=> %s ORDER BY id DESC LIMIT %s",
                        (session_id, limit),
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


def add_artifact(session_id, kind, label, content):
    """Record a generated artifact — a file write_file actually put on
    disk (kind='file'), or a code block the model showed without writing
    anywhere (kind='code'). Called from agent/core.py so both the CLI and
    GUI populate the same history."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO artifacts (session_id, kind, label, content) VALUES (%s, %s, %s, %s)",
                    (session_id, kind, label, content),
                )
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to save artifact: {exc}")


def list_artifacts(session_id, limit=50):
    """Return this thread's artifacts, newest first."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, kind, label, content, created_at FROM artifacts "
                    "WHERE session_id = %s ORDER BY id DESC LIMIT %s",
                    (session_id, limit),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {"id": id_, "kind": kind, "label": label, "content": content, "created_at": created_at}
            for id_, kind, label, content, created_at in rows
        ]
    except Exception as exc:
        print(f"[memory] failed to list artifacts: {exc}")
        return []


LESSON_STATUSES = {"pending", "active", "disabled", "quarantined", "rejected"}
LESSON_ORIGINS = {
    "owner_feedback", "participant_feedback", "verified_recovery",
    "contract_violation", "legacy", "legacy_auto",
}
LESSON_SCOPE_KINDS = {"global", "workspace", "channel", "tool"}

_LESSON_SELECT = (
    "id, keywords, lesson, source_session_id, hit_count, created_at, "
    "status, origin, scope_kind, scope_value, detector, fingerprint, "
    "occurrence_count, success_count, failure_count, "
    "consecutive_failure_count, last_seen_at, last_used_at, updated_at, "
    "merged_into_id"
)


def _lesson_dict(row):
    if not row:
        return None
    keys = (
        "id", "keywords", "lesson", "source_session_id", "hit_count",
        "created_at", "status", "origin", "scope_kind", "scope_value",
        "detector", "fingerprint", "occurrence_count", "success_count",
        "failure_count", "consecutive_failure_count", "last_seen_at",
        "last_used_at", "updated_at", "merged_into_id",
    )
    return dict(zip(keys, row))


def _normalize_lesson_fields(keywords, lesson):
    terms = []
    seen = set()
    for raw in (keywords or "").split(","):
        term = re.sub(r"\s+", " ", raw.strip().lower())
        if not term or term in seen:
            continue
        if len(term) > 64:
            term = term[:64].rstrip()
        terms.append(term)
        seen.add(term)
        if len(terms) == 8:
            break
    normalized_lesson = re.sub(r"\s+", " ", (lesson or "").strip())
    if not terms:
        raise ValueError("A lesson needs at least one trigger keyword or phrase.")
    if len(normalized_lesson) < 12:
        raise ValueError("A lesson must describe the corrected behavior.")
    return ",".join(terms), normalized_lesson[:2000]


def lesson_fingerprint(kind, signature):
    normalized = re.sub(r"\s+", " ", (signature or "").strip().lower())
    return hashlib.sha256(f"{kind}:{normalized}".encode("utf-8")).hexdigest()


def _insert_lesson_event(cur, lesson_id, event_kind, detector=None,
                         source_session_id=None, source_channel=None,
                         source_actor=None, evidence=None):
    cur.execute(
        "INSERT INTO lesson_events (lesson_id, event_kind, detector, "
        "source_session_id, source_channel, source_actor, evidence) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (
            lesson_id, event_kind, detector, source_session_id,
            source_channel, (source_actor or "")[:255] or None,
            (evidence or "")[:8000] or None,
        ),
    )


def upsert_lesson(keywords, lesson, *, status="pending", origin="owner_feedback",
                  scope_kind="global", scope_value=None, detector=None,
                  fingerprint=None, source_session_id=None,
                  source_channel=None, source_actor=None, evidence=None,
                  event_kind="observed"):
    """Create or reinforce one learning item and return its current record.

    A rejected/disabled/quarantined fingerprint is deliberately never
    reactivated by observation alone. A later explicit owner review can do
    that through update_lesson().
    """
    if status not in LESSON_STATUSES:
        raise ValueError(f"Invalid lesson status: {status}")
    if origin not in LESSON_ORIGINS:
        raise ValueError(f"Invalid lesson origin: {origin}")
    if scope_kind not in LESSON_SCOPE_KINDS:
        raise ValueError(f"Invalid lesson scope: {scope_kind}")
    keywords, lesson = _normalize_lesson_fields(keywords, lesson)
    if scope_kind == "global":
        scope_value = None
    elif not (scope_value or "").strip():
        raise ValueError(f"{scope_kind} lessons require a scope value")
    scope_value = (scope_value or "").strip() or None
    fingerprint = fingerprint or lesson_fingerprint(
        "lesson", f"{scope_kind}|{scope_value or ''}|{keywords}|{lesson}"
    )

    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_LESSON_SELECT} FROM lessons WHERE fingerprint = %s",
                    (fingerprint,),
                )
                existing = _lesson_dict(cur.fetchone())
                if existing:
                    # A rejected source row can remain as a fingerprint
                    # alias after a reviewer merges duplicates. Route new
                    # observations to the surviving canonical lesson so the
                    # merge remains effective instead of splitting again.
                    if existing.get("merged_into_id"):
                        cur.execute(
                            f"SELECT {_LESSON_SELECT} FROM lessons WHERE id = %s",
                            (existing["merged_into_id"],),
                        )
                        canonical = _lesson_dict(cur.fetchone())
                        if canonical:
                            existing = canonical
                    promote = existing["status"] == "pending" and status == "active"
                    cur.execute(
                        "UPDATE lessons SET occurrence_count = occurrence_count + 1, "
                        "last_seen_at = CURRENT_TIMESTAMP, status = %s WHERE id = %s",
                        ("active" if promote else existing["status"], existing["id"]),
                    )
                    if existing["status"] != "rejected":
                        _insert_lesson_event(
                            cur, existing["id"], event_kind, detector,
                            source_session_id, source_channel, source_actor, evidence,
                        )
                    cur.execute(
                        f"SELECT {_LESSON_SELECT} FROM lessons WHERE id = %s",
                        (existing["id"],),
                    )
                    record = _lesson_dict(cur.fetchone())
                    record["created_new"] = False
                    return record

                cur.execute(
                    "INSERT INTO lessons (keywords, lesson, source_session_id, "
                    "status, origin, scope_kind, scope_value, detector, "
                    "fingerprint, last_seen_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)",
                    (
                        keywords, lesson, source_session_id, status, origin,
                        scope_kind, scope_value, detector, fingerprint,
                    ),
                )
                lesson_id = cur.lastrowid
                _insert_lesson_event(
                    cur, lesson_id, event_kind, detector, source_session_id,
                    source_channel, source_actor, evidence,
                )
                cur.execute(
                    f"SELECT {_LESSON_SELECT} FROM lessons WHERE id = %s",
                    (lesson_id,),
                )
                record = _lesson_dict(cur.fetchone())
                record["created_new"] = True
                return record
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to upsert lesson: {exc}")
        return None


def get_lesson(lesson_id):
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_LESSON_SELECT} FROM lessons WHERE id = %s",
                    (lesson_id,),
                )
                return _lesson_dict(cur.fetchone())
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to get lesson: {exc}")
        return None


def list_lessons(status=None, include_rejected=False):
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                clauses = []
                params = []
                if status:
                    clauses.append("status = %s")
                    params.append(status)
                elif not include_rejected:
                    clauses.append("status <> 'rejected'")
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                cur.execute(
                    f"SELECT {_LESSON_SELECT} FROM lessons {where} "
                    "ORDER BY FIELD(status, 'pending', 'quarantined', 'active', 'disabled', 'rejected'), "
                    "updated_at DESC, id DESC",
                    params,
                )
                return [_lesson_dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to list lessons: {exc}")
        return []


def list_lesson_events(lesson_id, limit=25):
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, event_kind, detector, source_session_id, "
                    "source_channel, source_actor, evidence, created_at "
                    "FROM lesson_events WHERE lesson_id = %s "
                    "ORDER BY id DESC LIMIT %s",
                    (lesson_id, limit),
                )
                keys = (
                    "id", "event_kind", "detector", "source_session_id",
                    "source_channel", "source_actor", "evidence", "created_at",
                )
                return [dict(zip(keys, row)) for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to list lesson events: {exc}")
        return []


def update_lesson(lesson_id, *, keywords=None, lesson=None, status=None,
                  scope_kind=None, scope_value=None):
    current = get_lesson(lesson_id)
    if current is None:
        return None
    next_keywords = current["keywords"] if keywords is None else keywords
    next_lesson = current["lesson"] if lesson is None else lesson
    next_keywords, next_lesson = _normalize_lesson_fields(next_keywords, next_lesson)
    next_status = current["status"] if status is None else status
    next_scope_kind = current["scope_kind"] if scope_kind is None else scope_kind
    next_scope_value = current["scope_value"] if scope_kind is None and scope_value is None else scope_value
    if next_status not in LESSON_STATUSES or next_status == "rejected":
        raise ValueError("Use reject_lesson() for rejection; otherwise choose a valid status.")
    if next_scope_kind not in LESSON_SCOPE_KINDS:
        raise ValueError(f"Invalid lesson scope: {next_scope_kind}")
    if next_scope_kind == "global":
        next_scope_value = None
    elif not (next_scope_value or "").strip():
        raise ValueError(f"{next_scope_kind} lessons require a scope value")
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE lessons SET keywords = %s, lesson = %s, status = %s, "
                    "scope_kind = %s, scope_value = %s, "
                    "consecutive_failure_count = CASE WHEN %s = 'active' THEN 0 "
                    "ELSE consecutive_failure_count END WHERE id = %s",
                    (
                        next_keywords, next_lesson, next_status,
                        next_scope_kind, next_scope_value, next_status, lesson_id,
                    ),
                )
                _insert_lesson_event(cur, lesson_id, "reviewed")
        finally:
            conn.close()
        return get_lesson(lesson_id)
    except Exception as exc:
        print(f"[memory] failed to update lesson: {exc}")
        return None


def reject_lesson(lesson_id):
    """Retain the fingerprint while removing the proposal and evidence."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE lessons SET merged_into_id = NULL WHERE merged_into_id = %s",
                    (lesson_id,),
                )
                cur.execute("DELETE FROM lesson_events WHERE lesson_id = %s", (lesson_id,))
                cur.execute("DELETE FROM lesson_uses WHERE lesson_id = %s", (lesson_id,))
                cur.execute(
                    "UPDATE lessons SET status = 'rejected', keywords = '', "
                    "lesson = '', source_session_id = NULL, scope_value = NULL, "
                    "merged_into_id = NULL "
                    "WHERE id = %s",
                    (lesson_id,),
                )
        finally:
            conn.close()
        return True
    except Exception as exc:
        print(f"[memory] failed to reject lesson: {exc}")
        return False


def delete_lesson(lesson_id):
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE lessons SET merged_into_id = NULL WHERE merged_into_id = %s",
                    (lesson_id,),
                )
                cur.execute("DELETE FROM lesson_events WHERE lesson_id = %s", (lesson_id,))
                cur.execute("DELETE FROM lesson_uses WHERE lesson_id = %s", (lesson_id,))
                cur.execute("DELETE FROM lessons WHERE id = %s", (lesson_id,))
                return cur.rowcount > 0
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to delete lesson: {exc}")
        return False


def merge_lessons(source_id, target_id):
    if source_id == target_id:
        return False
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT occurrence_count, hit_count, success_count, failure_count "
                    "FROM lessons WHERE id = %s",
                    (source_id,),
                )
                source = cur.fetchone()
                cur.execute("SELECT id FROM lessons WHERE id = %s", (target_id,))
                if not source or not cur.fetchone():
                    return False
                cur.execute(
                    "UPDATE lessons SET occurrence_count = occurrence_count + %s, "
                    "hit_count = hit_count + %s, success_count = success_count + %s, "
                    "failure_count = failure_count + %s WHERE id = %s",
                    (*source, target_id),
                )
                cur.execute(
                    "UPDATE lesson_events SET lesson_id = %s WHERE lesson_id = %s",
                    (target_id, source_id),
                )
                cur.execute(
                    "UPDATE lesson_uses SET lesson_id = %s WHERE lesson_id = %s",
                    (target_id, source_id),
                )
                cur.execute(
                    "UPDATE lessons SET merged_into_id = %s WHERE merged_into_id = %s",
                    (target_id, source_id),
                )
                cur.execute(
                    "UPDATE lessons SET status = 'rejected', keywords = '', lesson = '', "
                    "source_session_id = NULL, scope_value = NULL, merged_into_id = %s "
                    "WHERE id = %s",
                    (target_id, source_id),
                )
                _insert_lesson_event(cur, target_id, "merged")
                return True
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to merge lessons: {exc}")
        return False


def quarantine_latest_feedback_lesson(session_id):
    """Undo the newest chat-taught lesson from this same conversation."""
    if session_id is None:
        return None
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT l.id FROM lessons l "
                    "JOIN lesson_events e ON e.lesson_id = l.id "
                    "WHERE e.source_session_id = %s "
                    "AND l.origin IN ('owner_feedback', 'participant_feedback') "
                    "AND l.status IN ('active', 'pending') "
                    "ORDER BY l.id DESC LIMIT 1",
                    (session_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                lesson_id = row[0]
                cur.execute(
                    "UPDATE lessons SET status = 'quarantined' WHERE id = %s",
                    (lesson_id,),
                )
                _insert_lesson_event(
                    cur, lesson_id, "owner_rollback",
                    source_session_id=session_id,
                )
                return lesson_id
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to quarantine latest feedback lesson: {exc}")
        return None


def _lesson_term_matches(text_lower, term):
    return bool(re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", text_lower))


def select_matching_lesson_records(records, text, *, workspace=None,
                                   channel=None, available_tools=None,
                                   limit=3):
    """Pure filtering/ranking half of match_lesson_records()."""
    workspace = os.path.abspath(workspace) if workspace else None
    available_tools = set(available_tools or [])
    text_lower = text.lower()
    scored = []
    for record in records:
        if record.get("status") != "active":
            continue
        scope_kind = record["scope_kind"]
        scope_value = record["scope_value"]
        if scope_kind == "workspace" and (
            not workspace or os.path.abspath(scope_value or "") != workspace
        ):
            continue
        if scope_kind == "channel" and scope_value != channel:
            continue
        if scope_kind == "tool" and scope_value not in available_tools:
            continue
        terms = [term.strip() for term in record["keywords"].split(",") if term.strip()]
        matched = [term for term in terms if _lesson_term_matches(text_lower, term)]
        if not matched:
            continue
        score = len(matched) * 100 + sum(min(len(term), 40) for term in matched)
        if scope_kind != "global":
            score += 20
        scored.append((score, record))

    return [record for _score, record in sorted(
        scored,
        key=lambda item: (item[0], item[1].get("updated_at") or item[1].get("id", 0)),
        reverse=True,
    )[:max(0, limit)]]


def match_lesson_records(text, *, workspace=None, channel=None,
                         available_tools=None, limit=3):
    """Return the highest-specificity active lessons valid in this context."""
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(f"SELECT {_LESSON_SELECT} FROM lessons WHERE status = 'active'")
                rows = [_lesson_dict(row) for row in cur.fetchall()]

                selected = select_matching_lesson_records(
                    rows, text, workspace=workspace, channel=channel,
                    available_tools=available_tools, limit=limit,
                )
                if selected:
                    cur.executemany(
                        "UPDATE lessons SET hit_count = hit_count + 1, "
                        "last_used_at = CURRENT_TIMESTAMP WHERE id = %s",
                        [(record["id"],) for record in selected],
                    )
                return selected
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to match lessons: {exc}")
        return []


def record_lesson_use(lesson_id, session_id=None, channel=None, detector=None):
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO lesson_uses (lesson_id, session_id, channel, detector) "
                    "VALUES (%s, %s, %s, %s)",
                    (lesson_id, session_id, channel, detector),
                )
                return cur.lastrowid
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to record lesson use: {exc}")
        return None


def resolve_lesson_use(use_id, outcome, failure_signature=None):
    if outcome not in {"success", "failure", "unknown"}:
        raise ValueError(f"Invalid lesson outcome: {outcome}")
    try:
        conn = _connect()
        try:
            _ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT lesson_id FROM lesson_uses WHERE id = %s", (use_id,))
                row = cur.fetchone()
                if not row:
                    return None
                lesson_id = row[0]
                cur.execute(
                    "UPDATE lesson_uses SET outcome = %s, failure_signature = %s, "
                    "resolved_at = CURRENT_TIMESTAMP WHERE id = %s",
                    (outcome, (failure_signature or "")[:255] or None, use_id),
                )
                if outcome == "success":
                    cur.execute(
                        "UPDATE lessons SET success_count = success_count + 1, "
                        "consecutive_failure_count = 0 WHERE id = %s",
                        (lesson_id,),
                    )
                elif outcome == "failure":
                    cur.execute(
                        "UPDATE lessons SET failure_count = failure_count + 1, "
                        "consecutive_failure_count = consecutive_failure_count + 1 "
                        "WHERE id = %s",
                        (lesson_id,),
                    )
                    cur.execute(
                        "UPDATE lessons SET status = 'quarantined' "
                        "WHERE id = %s AND origin IN ('verified_recovery', 'contract_violation') "
                        "AND consecutive_failure_count >= 2",
                        (lesson_id,),
                    )
                return get_lesson(lesson_id)
        finally:
            conn.close()
    except Exception as exc:
        print(f"[memory] failed to resolve lesson use: {exc}")
        return None


def add_lesson(keywords, lesson, source_session_id=None):
    """Backward-compatible reviewed/global lesson entry point."""
    record = upsert_lesson(
        keywords, lesson, status="active", origin="legacy",
        source_session_id=source_session_id, source_channel="legacy",
        event_kind="legacy_add",
    )
    return record["id"] if record else None


def match_lessons(text, **context):
    return [record["lesson"] for record in match_lesson_records(text, **context)]


def add_auto_lesson(keywords, lesson, detector, source_session_id=None):
    """Compatibility shim: unresolved legacy auto-lessons require review."""
    record = upsert_lesson(
        keywords, lesson, status="pending", origin="legacy_auto",
        scope_kind="tool", scope_value="run_shell_command", detector=detector,
        fingerprint=lesson_fingerprint("legacy-auto", f"{detector}|{keywords}|{lesson}"),
        source_session_id=source_session_id, source_channel="legacy",
        event_kind="legacy_auto",
    )
    return record["id"] if record else None


def match_auto_lessons(_text):
    """Legacy compatibility; migrated records are matched through lessons."""
    return []
