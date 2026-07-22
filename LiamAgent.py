#!/usr/bin/env python3
"""Command-line entry point for the Liam agent."""

import argparse
import os
import sys


def _load_dotenv(path=".env"):
    """Populate os.environ from a local KEY=VALUE file, if present. Real
    environment variables always take precedence over the file."""
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


def main():
    parser = argparse.ArgumentParser(description="Liam - a local CLI agent on Ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument(
        "--confirm", action="store_true",
        help="Prompt for approval before write_file/run_shell_command calls "
             "(default: run tools without asking)",
    )
    args = parser.parse_args()

    # prompt_toolkit handles bracketed paste correctly: a pasted multi-line
    # block gets inserted as literal text (newlines included), and only a
    # real Enter press submits it — unlike plain input(), which treats each
    # newline in a paste as its own Enter, submitting one message per line.
    # It requires a real terminal, though, so piped/non-interactive input
    # (scripts, tests) falls back to plain input() instead.
    prompt_fn = _make_prompt_fn()

    agent = Agent(model=args.model, auto_confirm=not args.confirm)
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
