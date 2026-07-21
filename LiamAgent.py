#!/usr/bin/env python3
"""Command-line entry point for the Liam agent."""

import argparse

from agent.core import Agent
from agent.llm import DEFAULT_MODEL


def main():
    parser = argparse.ArgumentParser(description="Liam - a local CLI agent on Ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument(
        "--confirm", action="store_true",
        help="Prompt for approval before write_file/run_shell_command calls "
             "(default: run tools without asking)",
    )
    args = parser.parse_args()

    agent = Agent(model=args.model, auto_confirm=not args.confirm)
    print(f"Liam agent ready (model: {args.model}). Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        reply = agent.step(user_input)
        print(f"\nliam> {reply}\n")


if __name__ == "__main__":
    main()
