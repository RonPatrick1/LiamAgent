"""Minimal local HTTP wrapper around Agent, for the Patrick Messenger
Matrix bot (liam_bot, a separate Rust service in another repo) to call
instead of hitting Ollama directly. This is what gives Matrix
conversations the real agent — tools, persistent memory, threads —
instead of the single-shot Q&A the Rust bot does on its own today.

Each Matrix room is its own isolated "bucket": its own message history,
and (unless it's the one designated shared-notes room) its own separate
notes pool, walled off from every other bucket's. Tool access is gated by
*sender* identity, not by room — only the configured owner ever gets
filesystem/shell access, wherever they are; everyone else gets a safe
subset regardless of which room they're in, even a room the owner is
also part of.

Binds to localhost by default. This must only ever be reachable from the
trusted liam_bot service on the same host/private network — never expose
it publicly. There's no authentication on this endpoint itself;
matrix-sdk on the Rust side is what actually verifies who really sent
each message (via Matrix's own server-side auth) before anything reaches
here, so room_id/sender_id are trusted inputs from that one caller, not
from the public internet.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import memory, routines
from . import tools as tools_module
from .core import Agent, ensure_visible_reply
from .contracts import PersistentActionContractStore
from .llm import DEFAULT_MODEL
from .tools import DESKTOP_ONLY_TOOLS, TOOL_IMPL

OWNER_MATRIX_ID = os.environ.get("LIAM_OWNER_MATRIX_ID", "")
SHARED_NOTES_ROOM_ID = os.environ.get("LIAM_SHARED_NOTES_ROOM_ID", "")
MESSENGER_WORKDIR = os.path.expanduser(os.environ.get("LIAM_MESSENGER_WORKDIR", "~/liam-messenger"))
FREDPLAYER_ASK_WORKDIR = os.path.expanduser(
    os.environ.get("LIAM_FREDPLAYER_ASK_WORKDIR", "~/liam-fredplayer")
)
HTTP_HOST = os.environ.get("LIAM_HTTP_HOST", "127.0.0.1")
HTTP_PORT = int(os.environ.get("LIAM_HTTP_PORT", "8787"))
MODEL = os.environ.get("LIAM_MESSENGER_MODEL") or DEFAULT_MODEL

# Everything except real filesystem/shell access — derived from the real
# tool registry rather than a hand-copied list, so a new tool added to
# tools.py later is excluded here by default instead of silently getting
# full access to every sender until someone remembers to update this too.
# propose_lesson remains restricted for defense-in-depth compatibility,
# though Agent no longer advertises it. Chat feedback now flows through
# the provenance-aware host classifier below instead.
RESTRICTED_TOOLS = {
    "read_file", "write_file", "edit_file", "list_directory", "run_shell_command", "propose_lesson",
    # Same filesystem/repo access tier as read_file/write_file above —
    # these all read or mutate the local machine's files, just like them.
    "search_text", "find_files", "file_info", "diff_files", "read_json",
    "make_directory", "copy_path", "move_path", "delete_path",
    "git_status", "git_diff", "git_log", "git_blame", "git_add",
}
SAFE_TOOLS = set(TOOL_IMPL) - RESTRICTED_TOOLS - DESKTOP_ONLY_TOOLS


def handle_chat(room_id, sender_id, message):
    room_id = (room_id or "").strip()
    sender_id = (sender_id or "").strip()
    message = (message or "").strip()
    if not room_id or not sender_id or not message:
        raise ValueError("room_id, sender_id, and message are all required")

    # get_or_create_session runs os.path.abspath() on this identifier
    # (it's normally a real folder path) — a leading slash keeps it
    # already-absolute so abspath() leaves it alone; without it, a plain
    # "matrix-room:X" string got the server's launch directory silently
    # prepended, which would fork a room onto a new session if the
    # server were ever started from a different cwd (proven, not
    # theoretical — caught by inspecting the actual row this wrote).
    session_id = memory.get_or_create_session(f"/matrix-room/{room_id}", title=f"Matrix: {room_id}")
    is_owner = bool(OWNER_MATRIX_ID) and sender_id == OWNER_MATRIX_ID
    notes_session_id = None if room_id == SHARED_NOTES_ROOM_ID else session_id

    os.makedirs(MESSENGER_WORKDIR, exist_ok=True)
    agent = Agent(
        model=MODEL,
        # No UI exists to show a confirmation dialog from a Matrix
        # message — same tradeoff already accepted for scheduled
        # routines, just reachable from chat instead of a timer. Applies
        # to every offered mutating tool. Lessons are handled separately
        # by the host-side feedback and evidence pipeline.
        auto_confirm=True,
        workdir=MESSENGER_WORKDIR,
        session_id=session_id,
        notes_session_id=notes_session_id,
        allowed_tools=None if is_owner else SAFE_TOOLS,
        channel="matrix", actor_id=sender_id, is_owner=is_owner,
        learning_enabled=True,
        action_contract_store=PersistentActionContractStore(),
    )
    return ensure_visible_reply(
        agent.step(message), stage="answering the Patrick Messenger request",
        tool_events=agent._tool_events,
    )


def handle_fredplayer_ask(device_id, message):
    """Backs the FredPlayer apps' in-app "Ask Liam" button. Reached only
    through the FredPlayer media server's own authenticated relay
    (POST /api/ask-liam on that server, itself only reachable over the
    nginx-fronted HTTPS path with the FredPlayer bearer token) — never
    called directly by a phone/device, so this endpoint itself stays
    unauthenticated same as /chat, on the same localhost-only trust
    boundary. Always SAFE_TOOLS, never owner tier — device_id is an
    unverified client-supplied string, nothing like Matrix's
    server-verified sender identity, so it must never be treated as an
    owner credential.

    Each device gets its own isolated session bucket (thread history +
    notes), same isolation model as each Matrix room today. Any playlist
    fredplayer_propose_playlist built during this turn is picked up here
    and returned directly — never written to disk, never visible to any
    other device.
    """
    device_id = (device_id or "").strip()
    message = (message or "").strip()
    if not device_id or not message:
        raise ValueError("device_id and message are both required")

    session_id = memory.get_or_create_session(
        f"/fredplayer-device/{device_id}", title=f"FredPlayer: {device_id}"
    )
    os.makedirs(FREDPLAYER_ASK_WORKDIR, exist_ok=True)
    agent = Agent(
        model=MODEL,
        auto_confirm=True,
        workdir=FREDPLAYER_ASK_WORKDIR,
        session_id=session_id,
        notes_session_id=session_id,
        allowed_tools=SAFE_TOOLS,
        channel="fredplayer", actor_id=device_id, is_owner=False,
        learning_enabled=False,
        action_contract_store=PersistentActionContractStore(),
        # Proven necessary: the local model would sometimes end its turn
        # with confident prose ("I've created a playlist called...")
        # without ever calling fredplayer_propose_playlist — the app then
        # gets playlist=None and silently does nothing, while the user
        # reads a reply that claims success. This is the same
        # narrate-instead-of-call failure mode DANGEROUS_TOOLS's
        # disallowed-tools warning was added for; same fix shape here.
        custom_instructions=(
            "When asked to build a playlist, reason about the request however "
            "you need to — mood, context, whatever's actually being asked — "
            "then you MUST actually call fredplayer_propose_playlist to hand "
            "it to the app; a playlist only reaches the user's device through "
            "that tool call, never through your reply text alone. Pick "
            "specific songs (artist + title), not whole artists — one "
            "artist's catalog can span very different moods, so naming "
            "individual tracks is what makes the playlist actually fit. You "
            "don't need exact file paths, matching happens automatically — "
            "call fredplayer_list_library(artist=...) on candidates first if "
            "you want to see their real track titles rather than guessing. "
            "Do not write a reply claiming you created, made, or saved a "
            "playlist unless you already called fredplayer_propose_playlist "
            "earlier in this same turn. If you could not finish, say so "
            "plainly instead of describing a playlist that doesn't exist."
        ),
    )
    # A failed turn (no fredplayer_propose_playlist call, so playlist is
    # still None) is auto-retried in the SAME conversation — the retry
    # prompt tells the model exactly what it skipped, which works better
    # than silently resending the identical original message. Capped at
    # 3 total attempts: retries cost real wall-clock time against a slow
    # local model, so this trades some latency for reliability rather
    # than retrying unboundedly.
    reply = ""
    playlist = None
    max_attempts = 3
    for attempt in range(max_attempts):
        prompt = message if attempt == 0 else (
            "That last turn ended without calling fredplayer_propose_playlist, "
            "so nothing was created — your reply text alone doesn't count. Try "
            "again: pick specific songs (artist + title) and call "
            "fredplayer_propose_playlist with them directly."
        )
        reply = ensure_visible_reply(
            agent.step(prompt), stage="answering the FredPlayer request",
            tool_events=agent._tool_events,
        )
        playlist = tools_module._PROPOSED_PLAYLISTS.pop(session_id, None)
        if playlist is not None:
            break
    if not reply and playlist is None:
        # Local models occasionally end a multi-tool-call turn with no
        # final text at all, especially after several fredplayer_list_library
        # calls in a row. Never hand the app back a literal blank string.
        reply = f"I tried {max_attempts} times but couldn't finish that — maybe rephrase with fewer artists at once."
    return {"reply": reply, "playlist": playlist}


class _ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # request-level logging belongs to the Rust side (tracing); keep this quiet

    def do_POST(self):
        if self.path not in (
            "/chat", "/fredplayer-ask",
            "/routine-deliveries/claim", "/routine-deliveries/ack",
        ):
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/chat":
                reply = handle_chat(body.get("room_id"), body.get("sender_id"), body.get("message"))
                payload = {"reply": reply}
            elif self.path == "/fredplayer-ask":
                payload = handle_fredplayer_ask(body.get("device_id"), body.get("message"))
            elif self.path == "/routine-deliveries/claim":
                payload = {"delivery": routines.claim_matrix_delivery()}
            else:
                delivery_id = int(body.get("id"))
                delivered = bool(body.get("delivered"))
                if not routines.resolve_matrix_delivery(
                    delivery_id, delivered, body.get("error"),
                ):
                    raise ValueError(f"routine delivery {delivery_id} does not exist")
                payload = {"ok": True}
            status = 200
        except Exception as exc:
            visible_error = (
                f"[error] Liam failed inside the {self.path} request handler "
                f"({type(exc).__name__}): {exc}"
            )
            # The Messenger/FredPlayer clients display `reply`; a bare HTTP
            # 400 with only an `error` field was discarded by callers and
            # looked exactly like Liam had ignored the user. Preserve the
            # protocol's normal response shape so the failure reaches the UI.
            if self.path == "/chat":
                payload = {"reply": visible_error, "error": str(exc)}
                status = 200
            elif self.path == "/fredplayer-ask":
                payload = {
                    "reply": visible_error,
                    "playlist": None,
                    "error": str(exc),
                }
                status = 200
            else:
                payload = {"error": str(exc)}
                status = 400

        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    if not OWNER_MATRIX_ID:
        print(
            "[server] WARNING: LIAM_OWNER_MATRIX_ID is not set — every "
            "sender will get the safe tool tier only, no one will get "
            "filesystem/shell access."
        )
    # The delivery bot polls frequently, so apply schema migrations once
    # at process startup instead of repeating DDL on every empty claim.
    conn = memory._connect()
    try:
        memory._ensure_schema(conn)
    finally:
        conn.close()
    server = ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), _ChatHandler)
    print(
        f"Liam messenger bridge listening on http://{HTTP_HOST}:{HTTP_PORT}/chat "
        f"and http://{HTTP_HOST}:{HTTP_PORT}/fredplayer-ask"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
