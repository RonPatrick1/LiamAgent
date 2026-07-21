"""Basic tool implementations and their JSON schemas."""

import os
import subprocess

import requests

WORKDIR = os.getcwd()

GOOGLE_SEARCH_API_KEY = os.environ.get("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_CX = os.environ.get("GOOGLE_SEARCH_CX")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")


def _resolve(path):
    return os.path.abspath(os.path.expanduser(path))


def read_file(path, max_chars=200_000):
    with open(_resolve(path), "r", errors="replace") as f:
        return f.read(max_chars)


def write_file(path, content):
    full_path = _resolve(path)
    parent = os.path.dirname(full_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {full_path}"


def list_directory(path="."):
    full_path = _resolve(path)
    return "\n".join(sorted(os.listdir(full_path)))


def run_shell_command(command, timeout=60):
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=timeout
    )
    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    output += f"\n[exit code: {result.returncode}]"
    return output[:20_000]


def _format_search_results(results):
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
    return "\n".join(lines)


def _google_search(query, num_results):
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_SEARCH_CX:
        raise RuntimeError("Google Search not configured (missing GOOGLE_SEARCH_API_KEY/GOOGLE_SEARCH_CX)")
    resp = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": GOOGLE_SEARCH_API_KEY,
            "cx": GOOGLE_SEARCH_CX,
            "q": query,
            "num": min(max(num_results, 1), 10),
        },
        timeout=15,
    )
    if resp.status_code in (403, 429):
        raise RuntimeError(f"Google Search quota/auth error ({resp.status_code}): {resp.text[:300]}")
    resp.raise_for_status()
    items = resp.json().get("items", [])
    return [
        {"title": i.get("title", ""), "url": i.get("link", ""), "snippet": i.get("snippet", "")}
        for i in items
    ]


def _brave_search(query, num_results):
    if not BRAVE_API_KEY:
        raise RuntimeError("Brave Search not configured (missing BRAVE_API_KEY)")
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
        params={"q": query, "count": min(max(num_results, 1), 10)},
        timeout=15,
    )
    resp.raise_for_status()
    items = resp.json().get("web", {}).get("results", [])
    return [
        {"title": i.get("title", ""), "url": i.get("url", ""), "snippet": i.get("description", "")}
        for i in items
    ]


def web_search(query, num_results=5):
    """Google Custom Search first; falls back to Brave if Google errors
    (quota exhausted, not configured, etc.)."""
    errors = []
    for name, fn in (("Google", _google_search), ("Brave", _brave_search)):
        try:
            return _format_search_results(fn(query, num_results))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    return "Web search failed on all backends.\n" + "\n".join(errors)


TOOL_IMPL = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "run_shell_command": run_shell_command,
    "web_search": web_search,
}

# Tools that mutate the filesystem or execute arbitrary commands require
# confirmation from the user before running (unless --yes is passed).
DANGEROUS_TOOLS = {"write_file", "run_shell_command"}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file from the local filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (or overwrite) a text file on the local filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write."},
                    "content": {"type": "string", "description": "Full text content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the entries of a directory on the local filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path. Defaults to the current directory."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell_command",
            "description": "Run a shell command on the local machine and return its stdout/stderr/exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Uses Google Custom "
                "Search, falling back to Brave Search if Google's quota is "
                "exhausted or unavailable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]
