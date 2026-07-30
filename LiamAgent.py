#!/usr/bin/env python3
"""Command-line entry point for the Liam agent."""

import argparse
import os
import subprocess
import sys


def _load_dotenv(path=None):
    """Populate os.environ from a local KEY=VALUE file, if present. Real
    environment variables always take precedence over the file.

    Resolved against this script's own directory, not the process's cwd —
    Liam can now be launched from any folder (each becomes its own
    session), and .env always lives alongside the code, not the folder a
    thread happens to be scoped to."""
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

from agent import memory, routines, settings as liam_settings
from agent.core import Agent, ensure_visible_reply
from agent.llm import DEFAULT_MODEL
from agent.tools import TOOL_IMPL, TOOL_SCHEMAS

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".liam_history")


def _make_prompt_fn():
    if sys.stdin.isatty():
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        session = PromptSession(history=FileHistory(HISTORY_FILE))
        return lambda: session.prompt("You>\n")

    import readline  # noqa: F401 (enables arrow-key history/editing where a real tty isn't in play but input() still applies)
    return lambda: input("You>\n")


def _help_text(tool_schemas=None):
    lines = ["REPL commands:", "  help          Show this help message", "  exit, quit    Quit Liam", "", "Tools Liam can use (it decides on its own when to call these):"]
    schemas = TOOL_SCHEMAS if tool_schemas is None else tool_schemas
    for schema in schemas:
        fn = schema["function"]
        desc = fn["description"].split(".")[0] + "."
        lines.append(f"  {fn['name']:<16} {desc}")
    return "\n".join(lines)


def _run_routine(routine_id):
    """Headless execution path for a scheduled routine — invoked by its
    systemd --user timer, not interactively. Runs the routine's prompt
    against its thread exactly like a normal launch would (same folder,
    extra folders, custom instructions), then queues encrypted delivery
    for a Matrix thread or uses notify-send for a desktop thread."""
    routine = routines.get_routine(routine_id)
    if routine is None:
        print(f"[routine] no such routine: {routine_id}")
        return

    session = memory.get_session(routine["session_id"])
    if session is None:
        print(f"[routine] routine {routine_id}'s thread no longer exists")
        return

    folder_path = session.get("folder_path") or ""
    matrix_prefix = "/matrix-room/"
    is_matrix = folder_path.startswith(matrix_prefix)
    workdir = (
        os.path.expanduser(os.environ.get("LIAM_MESSENGER_WORKDIR", "~/liam-messenger"))
        if is_matrix else folder_path
    )
    os.makedirs(workdir, exist_ok=True)

    saved = liam_settings.load()
    extra_folders = [f["folder_path"] for f in memory.list_session_folders(session["id"])]
    routine_allowed_tools = set(TOOL_IMPL) - {
        "schedule_routine", "cancel_routine", "list_my_routines",
    }
    try:
        agent = Agent(
            model=saved["model"] or DEFAULT_MODEL, auto_confirm=True,
            workdir=workdir, session_id=session["id"],
            extra_folders=extra_folders, custom_instructions=saved["custom_instructions"],
            channel="routine", actor_id="system-routine", is_owner=True,
            learning_enabled=False, allowed_tools=routine_allowed_tools,
        )
        reply = ensure_visible_reply(
            agent.step(routine["prompt"]), stage=f"running routine #{routine_id}",
            tool_events=agent._tool_events,
        )
    except Exception as exc:
        reply = (
            f"[error] Liam failed while running routine #{routine_id} "
            f"({type(exc).__name__}): {exc}"
        )

    if is_matrix:
        room_id = folder_path[len(matrix_prefix):]
        delivery_id = routines.enqueue_matrix_delivery(routine_id, room_id, reply)
        print(f"[routine] queued Matrix delivery #{delivery_id} for {room_id}")
    else:
        summary = reply.strip().splitlines()[0][:200] if reply.strip() else "(no reply)"
        try:
            completed = subprocess.run(
                ["notify-send", f"Liam routine: {session['title']}", summary],
                capture_output=True, timeout=5,
            )
            if completed.returncode != 0:
                detail = completed.stderr.decode(errors="replace").strip()
                print(
                    f"[routine] notify-send failed with exit code "
                    f"{completed.returncode}: {detail or '(no error text)'}"
                )
        except Exception as exc:
            print(f"[routine] notify-send failed: {exc}")

    routines.mark_ran(routine_id)
    if routine["schedule_kind"] == "once":
        routines.set_enabled(routine_id, False)


def main():
    saved = liam_settings.load()
    parser = argparse.ArgumentParser(description="Liam - a local CLI agent on Ollama")
    parser.add_argument("--model", default=saved["model"] or DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument(
        "--confirm", action="store_true", default=not saved["auto_confirm"],
        help="Prompt for approval before write_file/run_shell_command calls "
             "(default: run tools without asking)",
    )
    parser.add_argument(
        "--routine", type=int, metavar="ID",
        help="Run a scheduled routine headlessly (used by its systemd timer, not for interactive use) and exit",
    )
    args = parser.parse_args()

    if args.routine is not None:
        _run_routine(args.routine)
        return

    # prompt_toolkit handles bracketed paste correctly: a pasted multi-line
    # block gets inserted as literal text (newlines included), and only a
    # real Enter press submits it — unlike plain input(), which treats each
    # newline in a paste as its own Enter, submitting one message per line.
    # It requires a real terminal, though, so piped/non-interactive input
    # (scripts, tests) falls back to plain input() instead.
    prompt_fn = _make_prompt_fn()

    workdir = os.getcwd()
    session_id = memory.get_or_create_session(workdir)
    extra_folders = [f["folder_path"] for f in memory.list_session_folders(session_id)]
    agent = Agent(
        model=args.model, auto_confirm=not args.confirm, workdir=workdir, session_id=session_id,
        extra_folders=extra_folders, custom_instructions=saved["custom_instructions"],
        channel="cli", actor_id="local-owner", is_owner=True,
        learning_enabled=True,
    )
    print(f"Liam agent ready (model: {args.model}). Type 'help' for commands, 'exit' to quit.\n")

    while True:
        try:
            user_input = prompt_fn().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "help":
            print()
            print(_help_text(agent.tool_schemas))
            print()
            continue

        try:
            reply = ensure_visible_reply(
                agent.step(user_input), stage="processing the CLI request",
                tool_events=agent._tool_events,
            )
        except Exception as exc:
            reply = (
                f"[error] Liam failed while processing the CLI request "
                f"({type(exc).__name__}): {exc}"
            )
        print("\nLiam>")
        print(reply)
        print()


if __name__ == "__main__":
    main()
