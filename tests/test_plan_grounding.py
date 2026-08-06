import json
import os
import tempfile
import unittest
from unittest import mock

from agent import core


TEST_LISTENING_PORTS_RESULT = (
    "Current TCP listeners:\n"
    "- None found in /proc/net/tcp or /proc/net/tcp6\n"
    "Suggested currently-unused unprivileged TCP ports "
    "from 8000-8999: 8000, 8001\n"
    "These suggestions are absent from the current TCP listener "
    "table; availability must still be rechecked when the server starts."
)


class PlanGroundingTests(unittest.TestCase):
    def test_grounding_tools_do_not_replace_structured_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {
                "title": "Create Fluxa webpage",
                "objective": "Create a local webpage on port 8000.",
                "files": [
                    os.path.join(directory, "index.html"),
                ],
                "steps": [
                    "Create the listed webpage file.",
                    (
                        "Serve the directory with `nohup python3 -m "
                        "http.server 8000 --bind 0.0.0.0 "
                        ">/dev/null 2>&1 &` so it remains running during "
                        "validation."
                    ),
                ],
                "validation": [
                    {
                        "command": (
                            "test -f "
                            + os.path.join(directory, "index.html")
                        ),
                        "expected": "The webpage file exists.",
                    },
                ],
                "non_goals": [
                    "Do not modify unrelated files.",
                ],
                "risks": [
                    "Port availability may change before execution.",
                ],
            }

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "list_directory",
                                "arguments": {
                                    "path": directory,
                                },
                            },
                        },
                        {
                            "function": {
                                "name": "listening_ports",
                                "arguments": {},
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": (
                        "```liam-plan\n"
                        + json.dumps(plan)
                        + "\n```"
                    ),
                },
            ]

            with (
                mock.patch.object(
                    core,
                    "OllamaClient",
                    return_value=client,
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_notes",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_messages",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "match_lesson_records",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "save_message",
                ),
                mock.patch.object(
                    core.memory,
                    "get_latest_plan",
                    return_value=None,
                ),
                mock.patch.object(
                    core.memory,
                    "create_plan",
                    return_value=81,
                ) as create_plan,
                mock.patch.dict(
                    core.TOOL_IMPL,
                    {
                        "list_directory": (
                            lambda path, base_dir=None:
                            "index.html"
                        ),
                        "listening_ports": (
                            lambda **_kwargs:
                            TEST_LISTENING_PORTS_RESULT
                        ),
                    },
                ),
            ):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=True,
                    workdir=directory,
                    session_id=42,
                )
                agent.on_tool_call = mock.Mock()

                reply = agent.step(
                    "Make a plan to create a local webpage."
                )

            self.assertEqual(client.chat.call_count, 2)
            self.assertIn("```liam-plan", reply)
            self.assertIn(
                "[Plan draft #81 is ready for approval.]",
                reply,
            )
            create_plan.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
