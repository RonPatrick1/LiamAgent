# Tool-routing benchmark

`scripts/benchmark_tool_routing.py` replays prompts on which Liam previously
failed to select a real tool. It compares the normal full-prompt/full-catalog
decision with the focused recovery shape. Generated tool calls are parsed and
scored but never executed.

Run the installed comparison:

```bash
python3 scripts/benchmark_tool_routing.py \
  --models liam-mistral-small3.2:latest llama3.1:8b \
  --router-model llama3.1:8b
```

Use `--router-url` for Alien's Ollama endpoint, `--runs N` for repeated trials,
and `--json-out PATH` for machine-readable results. SSH cases are included in
normal primary-model scoring, but not generic recovery scoring: explicit remote
commands remain behind Liam's deterministic desktop-only SSH security boundary.

## 2026-07-30 result

| Model/role | Shape | Passed | Average |
|---|---:|---:|---:|
| `liam-mistral-small3.2:latest` | normal primary | 6/6 | 4.54s |
| `liam-mistral-small3.2:latest` | focused recovery | 4/4 | 1.78s |
| `llama3.1:8b` | normal primary | 4/6 | 1.09s |
| `llama3.1:8b` | focused recovery | 3/4 | 0.94s |
| local warmed `llama3.1:8b` | single-tool router | 4/4 | 0.40s |
| Alien GTX 1080 `llama3.1:8b` | single-tool router | 4/4 | 9.20s |

Mistral remains the primary model. Llama is substantially faster and reliable
as the constrained single-tool router, but it is not a safe primary replacement:
it selected the local `run_shell_command` tool for both remote-host prompts and
emitted three duplicate `read_file` calls in one focused case.

The configured Alien router is slower per request than a warmed local Llama on
this hardware. It is still used for this rare recovery-only step because it
does not unload or contend with Liam's 24B primary model on the desktop GPU.
