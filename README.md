# Liam Agent

A minimal CLI agent harness on top of `mistral-small3.2:24b`, served locally
via Ollama.

## Requirements

- Ollama running locally with `mistral-small3.2:24b` pulled (requires Ollama
  0.7+ for the `mistral3` architecture — this model won't load on older
  versions).
- `pip install -r requirements.txt`

## Run

```
python3 LiamAgent.py
```

Tool calls run without asking by default. Pass `--confirm` if you want to be
prompted before write_file/run_shell_command calls. Use `--model` to point at
a different Ollama model.

## GUI

`LiamGUI.py` is a native GTK4/libadwaita desktop app — same `Agent` class,
same tools, same memory, just a window instead of a terminal. Requires
`python3-gi` with GTK4 and libadwaita bindings (already present on a stock
Ubuntu GNOME desktop; not installable via pip).

```
python3 LiamGUI.py
```

Same `--model`/`--confirm` flags as the CLI. With `--confirm`, tool
confirmations show as a native dialog instead of a terminal prompt.

## Web search

`web_search` uses the Brave Search API (api-dashboard.search.brave.com).
Google Custom Search was considered but dropped — as of Jan 2026 Google no
longer allows newly created search engines to search the open web, only up
to 50 specified domains, which isn't useful as a general-purpose tool.

Set your key by copying `.env.example` to `.env` and editing it yourself:

```
cp .env.example .env
```

Then open `.env` in an editor and paste your key after `BRAVE_API_KEY=`.
`LiamAgent.py` loads `.env` automatically at startup (it's gitignored, so
the key never gets committed). If `BRAVE_API_KEY` isn't set, `web_search`
reports that plainly instead of failing silently.

## Persistent memory

Liam remembers past conversations across restarts via a MySQL table
(`messages`) on a fixed database server — host, port, and database name are
hardcoded in `agent/memory.py` (`192.168.0.136` / `liams_memory`), since
that's infrastructure, not something the agent should rediscover each run.
Only the credentials come from `.env`:

```
DB_USER=...
DB_PASSWORD=...
```

The `messages` table is created automatically on first use. Each user
message and Liam's final reply get saved; on startup, the last 20 messages
are loaded back into context. If the database is unreachable, memory calls
fail quietly (a `[memory] ...` warning is printed) rather than crashing the
agent — search and file tools still work fine without it.

## Layout

- `LiamAgent.py` — CLI entry point.
- `LiamGUI.py` — GTK4/libadwaita desktop entry point.
- `agent/llm.py` — thin client for Ollama's `/api/chat` endpoint.
- `agent/tools.py` — tool schemas and implementations (`read_file`,
  `write_file`, `list_directory`, `run_shell_command`, `web_search`).
- `agent/memory.py` — persistent conversation history backed by MySQL.
- `agent/core.py` — the agent loop: sends messages, executes tool calls the
  model requests, feeds results back, repeats until a final answer. UI-
  agnostic — `on_tool_call`/`on_confirm`/`on_status` callbacks let any
  frontend (CLI, GUI) hook into progress and confirmations without the
  agent itself depending on print()/input().
