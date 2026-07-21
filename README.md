# Liam Agent

A minimal CLI agent harness on top of `qwen2.5:32b-instruct`, served locally
via Ollama.

## Requirements

- Ollama running locally with `qwen2.5:32b-instruct` pulled.
- `pip install -r requirements.txt`

## Run

```
python3 LiamAgent.py
```

Tool calls run without asking by default. Pass `--confirm` if you want to be
prompted before write_file/run_shell_command calls. Use `--model` to point at
a different Ollama model.

## Web search

`web_search` tries Google Custom Search first, falling back to Brave Search
if Google errors out (daily quota exhausted, not configured, etc.). Set
whichever of these env vars you have before running `LiamAgent.py`:

```
export GOOGLE_SEARCH_API_KEY=...   # https://developers.google.com/custom-search/v1/introduction
export GOOGLE_SEARCH_CX=...        # your Custom Search Engine ID
export BRAVE_API_KEY=...           # https://api.search.brave.com/app/keys
```

Either provider can be left unset — if Google isn't configured it falls
through to Brave immediately; if both are unset, `web_search` reports that
plainly instead of failing silently.

## Layout

- `LiamAgent.py` — REPL entry point.
- `agent/llm.py` — thin client for Ollama's `/api/chat` endpoint.
- `agent/tools.py` — tool schemas and implementations (`read_file`,
  `write_file`, `list_directory`, `run_shell_command`, `web_search`).
- `agent/core.py` — the agent loop: sends messages, executes tool calls the
  model requests, feeds results back, repeats until a final answer.
