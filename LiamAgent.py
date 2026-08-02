#!/usr/bin/env python3
"""Command-line entry point for the Liam agent."""

import argparse
import os
import signal
import threading
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


def _resolve_cli_session(args):
    """Resolve the exact persisted GUI/CLI thread selected by the user."""
    if args.session_id is not None:
        session = memory.get_session(args.session_id)
        if session is None:
            raise LookupError(
                f"No Liam session exists with id {args.session_id}."
            )
        return session

    workdir = os.path.abspath(
        os.path.expanduser(args.workdir or os.getcwd())
    )
    session_id = memory.get_or_create_session(workdir)
    session = memory.get_session(session_id)

    if session is None:
        raise LookupError(
            f"Liam could not load the session for {workdir}."
        )

    return session


def _print_sessions():
    sessions = memory.list_sessions(include_archived=True)

    if not sessions:
        print("No Liam sessions found.")
        return

    for session in sessions:
        flags = []

        if session.get("plan_mode"):
            flags.append("plan")
        if session.get("archived"):
            flags.append("archived")
        if session.get("pinned"):
            flags.append("pinned")

        flag_text = ",".join(flags) if flags else "normal"
        print(
            f'{session["id"]}\t[{flag_text}]\t'
            f'{session["title"]}\t{session["folder_path"]}'
        )


def _run_cli_request(agent, user_input):
    try:
        reply = ensure_visible_reply(
            agent.step(user_input),
            stage="processing the CLI request",
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


def _approve_cli_plan(session_id, plan_id):
    """Explicitly approve one stored Plan for the selected session."""
    plan = memory.get_plan(plan_id)

    if not plan:
        return f"FAIL: plan #{plan_id} was not found."

    if plan.get("session_id") != session_id:
        return (
            f"FAIL: plan #{plan_id} belongs to a different thread."
        )

    status = plan.get("status")

    if status in {"draft", "failed"}:
        try:
            changed = memory.transition_plan(
                plan_id,
                status,
                "approved",
            )
        except Exception as exc:
            return (
                f"FAIL: plan #{plan_id} could not be approved "
                f"({type(exc).__name__}: {exc})."
            )

        if not changed:
            return (
                f"FAIL: plan #{plan_id} could not transition from "
                f"{status!r} to 'approved'."
            )
    elif status != "approved":
        return (
            f"FAIL: plan #{plan_id} has status {status!r} "
            "and cannot be approved and run."
        )

    try:
        memory.set_plan_mode(session_id, False)
    except Exception as exc:
        return (
            "FAIL: the session could not leave Plan mode before "
            f"execution ({type(exc).__name__}: {exc})."
        )

    return None


def _run_cli_plan(agent, plan_id):
    """Run the existing approved-Plan executor in the foreground."""
    cancel_event = threading.Event()
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_cancel(_signum, _frame):
        if cancel_event.is_set():
            return

        cancel_event.set()
        print(
            "\nCancellation requested. Liam will stop at the next "
            "safe checkpoint."
        )

    signal.signal(signal.SIGINT, request_cancel)

    try:
        try:
            reply = agent.execute_plan(
                plan_id,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            reply = (
                "FAIL: Liam crashed while executing the approved plan "
                f"({type(exc).__name__}: {exc})."
            )
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    reply = ensure_visible_reply(
        reply,
        stage="executing the approved plan",
    )

    print("\nLiam>")
    print(reply)
    print()


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
    session_selector = parser.add_mutually_exclusive_group()
    session_selector.add_argument(
        "--session-id",
        type=int,
        metavar="ID",
        help=(
            "Use the exact persisted Liam thread with this session ID, "
            "including its GUI history, folders, plans, and Plan-mode state."
        ),
    )
    session_selector.add_argument(
        "--workdir",
        metavar="PATH",
        help=(
            "Use the persisted Liam thread associated with this folder. "
            "Defaults to the current directory."
        ),
    )
    cli_action = parser.add_mutually_exclusive_group()
    cli_action.add_argument(
        "--prompt",
        metavar="TEXT",
        help=(
            "Process one request and exit instead of opening the "
            "interactive REPL."
        ),
    )
    cli_action.add_argument(
        "--run-plan",
        type=int,
        metavar="ID",
        help=(
            "Explicitly approve and immediately run this stored Plan "
            "in the selected session."
        ),
    )
    cli_action.add_argument(
        "--list-sessions",
        action="store_true",
        help=(
            "List persisted Liam session IDs, states, titles, and "
            "folder paths, then exit."
        ),
    )
    args = parser.parse_args()

    if args.routine is not None:
        _run_routine(args.routine)
        return

    if args.list_sessions:
        _print_sessions()
        return

    try:
        session = _resolve_cli_session(args)
    except LookupError as exc:
        print(f"[error] {exc}")
        return

    session_id = session["id"]
    workdir = session["folder_path"]

    if args.run_plan is not None:
        approval_error = _approve_cli_plan(
            session_id,
            args.run_plan,
        )

        if approval_error:
            print(approval_error)
            return

        session = dict(session)
        session["plan_mode"] = False

    extra_folders = [
        folder["folder_path"]
        for folder in memory.list_session_folders(session_id)
    ]
    agent = Agent(
        model=args.model,
        auto_confirm=not args.confirm,
        workdir=workdir,
        session_id=session_id,
        extra_folders=extra_folders,
        custom_instructions=saved["custom_instructions"],
        channel="cli",
        actor_id="local-owner",
        is_owner=True,
        learning_enabled=True,
        plan_mode=bool(session.get("plan_mode")),
    )
    print(
        f'Liam agent ready for session #{session_id} '
        f'("{session["title"]}", model: {args.model}, '
        f'Plan mode: {"on" if agent.plan_mode else "off"}).'
    )

    if args.run_plan is not None:
        _run_cli_plan(agent, args.run_plan)
        return

    if args.prompt is not None:
        user_input = args.prompt.strip()

        if not user_input:
            print("[error] --prompt must contain a non-empty request.")
            return

        _run_cli_request(agent, user_input)
        return

    print("Type 'help' for commands, 'exit' to quit.\n")

    # prompt_toolkit handles bracketed paste correctly: a pasted multi-line
    # block gets inserted as literal text (newlines included), and only a
    # real Enter press submits it — unlike plain input(), which treats each
    # newline in a paste as its own Enter, submitting one message per line.
    # It requires a real terminal, though, so piped/non-interactive input
    # (scripts, tests) falls back to plain input() instead.
    prompt_fn = _make_prompt_fn()

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

        _run_cli_request(agent, user_input)


if __name__ == "__main__":
    main()
