#!/usr/bin/env python3
"""Benchmark primary-model tool selection without executing any tools.

Each case is sent in two shapes: the normal full tool catalog and the focused
recovery shape (recent user requests only plus the case's relevant tools). The
script records only the model's first response, so it is safe to run: generated
tool calls are scored but never dispatched to TOOL_IMPL.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.core import (  # noqa: E402
    Agent,
    RECOVERY_SYSTEM_PROMPT,
    RECOVERY_USER_CONTEXT_LIMIT,
    RECOVERY_USER_MESSAGE_CHARS,
    SYSTEM_PROMPT,
    TOOL_DEFLECTION_RE,
)
from agent.llm import DEFAULT_MODEL, DEFAULT_URL, OllamaClient  # noqa: E402
from agent.tools import TOOL_SCHEMAS  # noqa: E402

DEFAULT_CASES = ROOT / "benchmarks" / "tool_routing_cases.json"


def offered_gui_tools():
    return [
        schema for schema in TOOL_SCHEMAS
        if schema["function"]["name"] != "propose_lesson"
    ]


def focused_user_text(messages):
    requests = [
        message.get("content", "").strip()[:RECOVERY_USER_MESSAGE_CHARS]
        for message in messages
        if message.get("role") == "user" and message.get("content", "").strip()
    ][-RECOVERY_USER_CONTEXT_LIMIT:]
    parts = []
    for index, content in enumerate(requests):
        label = "Current user request" if index == len(requests) - 1 else "Earlier user request"
        parts.append(f"{label}:\n{content}")
    return "\n\n".join(parts)


def response_tool_names(response):
    calls = response.get("tool_calls") if isinstance(response, dict) else None
    if calls is None:
        return [], None
    if not isinstance(calls, list):
        return [], "tool_calls is not a list"
    names = []
    for index, call in enumerate(calls, start=1):
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            return names, f"tool call #{index} has no function object"
        function = call["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            return names, f"tool call #{index} has no name"
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError) as exc:
                return names, f"tool call #{index} has invalid arguments: {exc}"
        if not isinstance(arguments, dict):
            return names, f"tool call #{index} arguments are not an object"
        names.append(name)
    return names, None


def score_response(case, response, offered_names):
    names, protocol_error = response_tool_names(response)
    expected = set(case["expected_tools"])
    content = response.get("content", "") if isinstance(response, dict) else ""
    quality_error = None
    max_tool_calls = int(case.get("max_tool_calls", 2))
    if len(names) > max_tool_calls:
        quality_error = (
            f"emitted {len(names)} tool calls; case limit is {max_tool_calls}"
        )
    passed = (
        protocol_error is None
        and quality_error is None
        and bool(expected.intersection(names))
        and all(name in offered_names for name in names)
    )
    return {
        "passed": passed,
        "tool_names": names,
        "protocol_error": protocol_error,
        "quality_error": quality_error,
        "deflected": bool(TOOL_DEFLECTION_RE.search(content or "")),
        "content": (content or "")[:500],
    }


def request_shape(case, mode, all_tools, recovery_names=None):
    if mode == "normal":
        return (
            [{"role": "system", "content": SYSTEM_PROMPT}] + case["messages"],
            all_tools,
        )
    relevant = set(case["relevant_tools"] if recovery_names is None else recovery_names)
    focused_tools = [
        schema for schema in all_tools
        if schema["function"]["name"] in relevant
    ]
    return (
        [
            {"role": "system", "content": RECOVERY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Recent user requests:\n{focused_user_text(case['messages'])}\n\n"
                    "Choose and call the appropriate listed tool now."
                ),
            },
        ],
        focused_tools,
    )


def route_recovery_tools(case, all_tools, model, url, timeout):
    agent = Agent.__new__(Agent)
    agent.tool_schemas = all_tools
    agent.messages = list(case["messages"])
    agent.on_status = lambda _message: None
    router = OllamaClient(
        model=model, url=url, timeout=timeout, keep_alive="30m",
        options={"temperature": 0}, response_format="json",
    )
    agent.client = router
    agent.helper_client = router
    started = time.monotonic()
    selected = agent._select_recovery_tool_schemas(len(agent.messages) - 1)
    elapsed = time.monotonic() - started
    names = [schema["function"]["name"] for schema in selected]
    expected = set(case["expected_tools"])
    relevant = set(case["relevant_tools"])
    passed = bool(expected.intersection(names)) and set(names).issubset(relevant)
    return names, {
        "model": f"router:{model}",
        "case": case["id"],
        "mode": "router",
        "run": 1,
        "seconds": round(elapsed, 3),
        "passed": passed,
        "tool_names": names,
        "protocol_error": None,
        "quality_error": None if passed else "selected tools were missing or outside the relevant set",
        "deflected": False,
        "content": "",
    }


def run_benchmark(
    models, url, cases, runs, timeout, router_model, router_url,
    router_only=False,
):
    all_tools = offered_gui_tools()
    results = []
    routed_names = {}
    for case in cases:
        if not case.get("recovery_eligible", True):
            continue
        names, result = route_recovery_tools(
            case, all_tools, router_model, router_url, timeout,
        )
        routed_names[case["id"]] = names
        results.append(result)
        state = "PASS" if result["passed"] else "FAIL"
        tools_label = ",".join(names) or "none"
        print(
            f"{state:4}  {result['model']:32}  router   {case['id']:24}  "
            f"{result['seconds']:6.2f}s  tools={tools_label}"
        )
    if router_only:
        return results
    for model in models:
        client = OllamaClient(model=model, url=url, timeout=timeout, keep_alive="30m")
        for case in cases:
            modes = ("normal", "focused") if case.get("recovery_eligible", True) else ("normal",)
            for mode in modes:
                messages, tools = request_shape(
                    case, mode, all_tools, routed_names.get(case["id"]),
                )
                offered_names = {schema["function"]["name"] for schema in tools}
                for run in range(1, runs + 1):
                    started = time.monotonic()
                    response = client.chat(messages, tools=tools)
                    elapsed = time.monotonic() - started
                    score = score_response(case, response, offered_names)
                    result = {
                        "model": model,
                        "case": case["id"],
                        "mode": mode,
                        "run": run,
                        "seconds": round(elapsed, 3),
                        **score,
                    }
                    results.append(result)
                    state = "PASS" if score["passed"] else "FAIL"
                    tools_label = ",".join(score["tool_names"]) or "none"
                    detail = (
                        score["protocol_error"] or score["quality_error"]
                        or score["content"].replace("\n", " ")[:100]
                    )
                    print(
                        f"{state:4}  {model:32}  {mode:7}  {case['id']:24}  "
                        f"{elapsed:6.2f}s  tools={tools_label}  {detail}"
                    )
    return results


def print_summary(results):
    print("\nSummary")
    keys = sorted({(item["model"], item["mode"]) for item in results})
    for model, mode in keys:
        group = [
            item for item in results
            if item["model"] == model and item["mode"] == mode
        ]
        passed = sum(item["passed"] for item in group)
        average = sum(item["seconds"] for item in group) / len(group)
        print(f"{model:32}  {mode:7}  {passed}/{len(group)} passed  avg {average:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Liam tool-call selection without executing tools.",
    )
    parser.add_argument(
        "--models", nargs="+", default=[DEFAULT_MODEL, "llama3.1:8b"],
        help="Ollama model names to compare.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="Ollama /api/chat URL.")
    parser.add_argument(
        "--router-model", default="llama3.1:8b",
        help="Preprocessing model used to select focused recovery tools.",
    )
    parser.add_argument(
        "--router-url", default=DEFAULT_URL,
        help="Ollama /api/chat URL for the preprocessing router.",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--router-only", action="store_true",
        help="Benchmark only relevant-tool selection, not primary model responses.",
    )
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text())
    results = run_benchmark(
        args.models, args.url, cases, max(1, args.runs), args.timeout,
        args.router_model, args.router_url, args.router_only,
    )
    print_summary(results)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
