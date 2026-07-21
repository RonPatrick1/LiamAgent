"""The agent loop: send messages to the model, execute any tool calls it
requests, feed results back, and repeat until it produces a final answer."""

import json

from .llm import OllamaClient, DEFAULT_MODEL
from .tools import TOOL_SCHEMAS, TOOL_IMPL, DANGEROUS_TOOLS

SYSTEM_PROMPT = """You are Liam, a local autonomous agent running on the \
qwen2.5:32b-instruct model via Ollama, operating through a command-line \
interface. You can use tools to read and write files, list directories, run \
shell commands on the user's machine, and search the web for current \
information.

Use tools when you need information you don't have or need to take action.
Reach for web_search for anything time-sensitive or beyond your training
data — don't guess. Don't call a tool speculatively if you can already
answer. When you're done, give a clear, concise final answer in plain text."""

MAX_STEPS = 10


class Agent:
    def __init__(self, model=DEFAULT_MODEL, auto_confirm=False):
        self.client = OllamaClient(model=model)
        self.auto_confirm = auto_confirm
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _confirm(self, name, args):
        if self.auto_confirm:
            return True
        print(f"\n[tool request] {name}({json.dumps(args)})")
        answer = input("Allow this? [y/N] ").strip().lower()
        return answer == "y"

    def _run_tool(self, name, args):
        if name not in TOOL_IMPL:
            return f"Error: unknown tool '{name}'"
        if name in DANGEROUS_TOOLS and not self._confirm(name, args):
            return "User denied this tool call."
        try:
            return str(TOOL_IMPL[name](**args))
        except Exception as exc:
            return f"Error: {exc}"

    def step(self, user_input):
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_STEPS):
            message = self.client.chat(self.messages, tools=TOOL_SCHEMAS)
            self.messages.append(message)

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return message.get("content", "")

            for call in tool_calls:
                fn = call["function"]
                name = fn["name"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args)

                print(f"  -> {name}({json.dumps(args)})")
                result = self._run_tool(name, args)
                self.messages.append({"role": "tool", "content": result})

        return "(stopped: reached max reasoning steps without a final answer)"
