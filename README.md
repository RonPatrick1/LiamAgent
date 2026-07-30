# Liam Agent

A minimal CLI agent harness on top of a local Liam-specific Ollama variant of
`mistral-small3.2:24b`.

## Requirements

- Ollama running locally with `mistral-small3.2:24b` pulled (requires Ollama
  0.7+ for the `mistral3` architecture — this model won't load on older
  versions).
- `pip install -r requirements.txt`

Create Liam's model once after pulling the upstream weights. This replaces
Mistral's bundled Le Chat capability prompt—which incorrectly says image
generation and web access are unavailable—while reusing the same local model
weights:

```
ollama create liam-mistral-small3.2:latest -f Modelfile.liam
```

## Run

```
python3 LiamAgent.py
```

### Optional remote helper model

Liam can offload hidden corrective-feedback classification, verified lesson
extraction, and relevance scanning of chunks from large file/web results to a
smaller Ollama model on another trusted LAN computer. This moves narrow
preprocessing calls off the local 24B model while the main answers, vision,
final synthesis, and tool decisions continue to use Liam's selected local
model unchanged.

```dotenv
LIAM_HELPER_OLLAMA_URL=http://192.168.0.128:11434/api/chat
LIAM_HELPER_OLLAMA_MODEL=llama3.1:8b
LIAM_HELPER_OLLAMA_TIMEOUT=45
LIAM_HELPER_OLLAMA_KEEP_ALIVE=30m
```

If the helper is unreachable, times out, or returns an invalid response, Liam
automatically repeats only that preprocessing step on its primary model. The
helper receives only the bounded input needed for its current job: the
immediately preceding answer and latest user message for feedback
classification, verified failure/recovery evidence for lesson extraction, or
one document chunk plus the current question for relevance scanning. It does
not receive Liam's tools or full conversation history.

The example points directly at Alien's LAN address. Alien's Ollama service must
set `OLLAMA_HOST=0.0.0.0:11434` so both Alien's local CLI and LAN clients can
reach it; no SSH tunnel is required.

Tool calls run without asking by default. Pass `--confirm` if you want to be
prompted before write_file/run_shell_command calls. Use `--model` to point at
a different Ollama model.

## GUI

`LiamGUI.py` is a native GTK3 desktop app — same `Agent` class,
same tools, same memory, just a window instead of a terminal. Requires
`python3-gi` with GTK3 bindings (already present on a stock
Ubuntu GNOME desktop; not installable via pip).

```
python3 LiamGUI.py
```

Same `--model`/`--confirm` flags as the CLI. With `--confirm`, tool
confirmations show as a native dialog instead of a terminal prompt.

## Desktop command history

Every non-empty desktop input is stored in a dedicated MySQL
`command_history` table. History is shared across Liam's desktop threads,
persists across restarts, and is capped at the newest 1,000 entries. It is an
editor feature and is never added to the model context merely because it was
recalled or listed.

- **Up / Ctrl+P** — previous entry
- **Down / Ctrl+N** — next entry, restoring the unfinished draft at the end
- **Alt+< / Alt+>** — oldest entry / newest unfinished draft
- **Alt+Up / Alt+Down** — previous / next entry matching the current prefix
- **Ctrl+R / Ctrl+S** — reverse / forward incremental substring search
- `history` or `history N` — show a numbered listing locally without asking
  the model (`history help` shows the shortcuts)

History navigation only fills the editor. It never executes a recalled entry;
the user must still press Enter or click Send.

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

## Desktop-only SSH

The Ubuntu desktop app can run non-interactive commands on explicitly
allowlisted SSH aliases. These tools are removed from the CLI, Patrick
Messenger, FredPlayer, and scheduled routines at both schema and execution
time.

Configure each computer normally in `~/.ssh/config`, using an SSH key and
`IdentityFile` (or `ssh-agent`), and connect manually once so its host key is
present in `~/.ssh/known_hosts`. Then add only the aliases Liam may use:

```dotenv
LIAM_SSH_HOSTS=worklaptop,jetson,alien
```

Liam uses batch-mode public-key authentication, refuses unknown host keys,
and rejects any host not in that list. Passwords in `.env` are intentionally
unsupported: they would expose reusable login secrets to the long-running app
and subprocess environment.

For commands that require administrator privileges, open **Customize → SSH
sudo passwords** in the Ubuntu desktop app and save the destination's sudo
password in GNOME Keyring. Liam can then call the same SSH tool with
`sudo: true`; the credential is looked up internally, passed only over the SSH
process's standard input for `sudo -S -v`, and redacted from returned output.
Removing a credential from Customize does not change the remote computer's
sudo configuration. Liam never creates passwordless-sudo or sudoers entries.

For an exact command, use the desktop form ``On HOST, run `COMMAND` with
sudo.`` Liam routes that form directly to `ssh_run_command` without asking the
model to reconstruct SSH or password handling. The generic local shell tool
rejects SSH clients and stdin-password sudo pipelines. When the request ends
in `with sudo`, the backticks are optional. The natural non-sudo form `On
HOST, run COMMAND.` is also routed directly; use backticks when terminal
punctuation is intentionally part of the literal command.

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

Raw tool protocol messages are discarded between turns. Large file/tool
results are reduced before they enter model-visible history, unrestricted
file reads return a bounded preview, and every Ollama request has a
conservative message budget. If Ollama still reports a context overflow,
Liam compacts further and retries once while preserving the real server error
if that retry also fails.

## Learning from corrections and failures

Lessons are separate from remembered user notes. Liam can activate a lesson
automatically only when the host has evidence it can check independently:

- a failed or no-op tool call is followed by a changed attempt that succeeds;
- a build, test, or lint command fails and a later run of the same validator
  exits successfully; or
- a fixed contract is violated, such as claiming an image was generated
  without calling the image tool, repeating an identical failed call, or
  reporting a failed action as successful.

Temporary service outages and denied tool calls do not produce lessons.
Duplicate observations reinforce one fingerprinted lesson instead of growing
the prompt indefinitely. Active lessons are keyword- and scope-matched, with
at most three injected into a turn. Their later outcomes are tracked; an
automatically created lesson is quarantined after two consecutive observed
failures.

Normal GUI, CLI, and Matrix chats can also provide corrective feedback. An
explicit, high-confidence correction from the configured owner becomes active
immediately. Ambiguous owner feedback and all non-owner feedback are queued as
pending candidates. Scheduled routines and FredPlayer device chats cannot
teach Liam. In an owner chat, `don't learn that` quarantines the newest lesson
taught from that conversation.

The desktop header's **Lessons** button opens the review queue. From there you
can inspect provenance/evidence and effectiveness counts, edit and approve a
candidate, disable/reactivate it, merge duplicates, reject it while retaining
its deduplication fingerprint, or delete it completely.

## Layout

- `LiamAgent.py` — CLI entry point.
- `LiamGUI.py` — GTK3 desktop entry point.
- `agent/llm.py` — thin client for Ollama's `/api/chat` endpoint.
- `agent/tools.py` — tool schemas and implementations (`read_file`,
  `write_file`, `list_directory`, `run_shell_command`, `web_search`).
- `agent/memory.py` — persistent conversation history backed by MySQL.
- `agent/core.py` — the agent loop: sends messages, executes tool calls the
  model requests, feeds results back, repeats until a final answer. UI-
  agnostic — `on_tool_call`/`on_confirm`/`on_status` callbacks let any
  frontend (CLI, GUI) hook into progress and confirmations without the
  agent itself depending on print()/input().
