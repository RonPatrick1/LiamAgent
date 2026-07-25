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
from agent.core import Agent
from agent.llm import DEFAULT_MODEL
from agent.tools import TOOL_SCHEMAS

HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".liam_history")


def _make_prompt_fn():
    if sys.stdin.isatty():
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        session = PromptSession(history=FileHistory(HISTORY_FILE))
        return lambda: session.prompt("You>\n")

    import readline  # noqa: F401 (enables arrow-key history/editing where a real tty isn't in play but input() still applies)
    return lambda: input("You>\n")


def _help_text():
    lines = ["REPL commands:", "  help          Show this help message", "  exit, quit    Quit Liam", "", "Tools Liam can use (it decides on its own when to call these):"]
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]
        desc = fn["description"].split(".")[0] + "."
        lines.append(f"  {fn['name']:<16} {desc}")
    return "\n".join(lines)


def _run_routine(routine_id):
    """Headless execution path for a scheduled routine — invoked by its
    systemd --user timer, not interactively. Runs the routine's prompt
    against its thread exactly like a normal launch would (same folder,
    extra folders, custom instructions), then notifies via notify-send
    since nothing is watching a terminal at 3am."""
    routine = routines.get_routine(routine_id)
    if routine is None:
        print(f"[routine] no such routine: {routine_id}")
        return

    session = memory.get_session(routine["session_id"])
    if session is None:
        print(f"[routine] routine {routine_id}'s thread no longer exists")
        return

    saved = liam_settings.load()
    extra_folders = [f["folder_path"] for f in memory.list_session_folders(session["id"])]
    agent = Agent(
        model=saved["model"] or DEFAULT_MODEL, auto_confirm=True,
        workdir=session["folder_path"], session_id=session["id"],
        extra_folders=extra_folders, custom_instructions=saved["custom_instructions"],
    )
    reply = agent.step(routine["prompt"])
    routines.mark_ran(routine_id)

    summary = reply.strip().splitlines()[0][:200] if reply.strip() else "(no reply)"
    try:
        subprocess.run(
            ["notify-send", f"Liam routine: {session['title']}", summary],
            capture_output=True, timeout=5,
        )
    except Exception as exc:
        print(f"[routine] notify-send failed: {exc}")


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
            print(_help_text())
            print()
            continue

        reply = agent.step(user_input)
        print("\nLiam>")
        print(reply)
        print()


if __name__ == "__main__":
    main()
