import json
import os
import tempfile
import unittest
from unittest import mock

from agent import core


def fenced(payload):
    return "```liam-plan\n" + json.dumps(payload) + "\n```"


TEST_LISTENING_PORTS_RESULT = (
    "Current TCP listeners:\n"
    "- None found in /proc/net/tcp or /proc/net/tcp6\n"
    "Suggested currently-unused unprivileged TCP ports "
    "from 8000-8999: 8000, 8001\n"
    "These suggestions are absent from the current TCP listener "
    "table; availability must still be rechecked when the server starts."
)


class PlanRecoveryLimitTests(unittest.TestCase):
    def test_second_bounded_plan_correction_can_succeed(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")

            base = {
                "title": "Create local webpage",
                "objective": "Create a local webpage at http://192.168.0.178:8000.",
                "files": [index_path],
                "validation": [
                    {
                        "command": "curl -fsS http://192.168.0.178:8000/ >/dev/null",
                        "expected": "The webpage responds successfully.",
                    }
                ],
                "non_goals": ["Do not modify system services or use sudo."],
                "risks": ["Port availability can change before execution."],
            }

            first = dict(base)
            first["steps"] = [
                f"Create {index_path}.",
                "Start a local web server on port 8000.",
            ]

            second = dict(base)
            second["steps"] = [
                f"Create {index_path}.",
                "Run `python3 -m http.server 8000 --bind 0.0.0.0`.",
            ]

            third = dict(base)
            third["steps"] = [
                f"Create {index_path}.",
                (
                    "Run `nohup python3 -m http.server 8000 "
                    "--bind 0.0.0.0 >/dev/null 2>&1 &` so it remains "
                    "running during validation."
                ),
            ]

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "listening_ports",
                            "arguments": {},
                        },
                    }],
                },
                {"role": "assistant", "content": fenced(first)},
                {"role": "assistant", "content": fenced(second)},
                {"role": "assistant", "content": fenced(third)},
            ]

            with (
                mock.patch.object(core, "OllamaClient", return_value=client),
                mock.patch.object(core.memory, "load_recent_notes", return_value=[]),
                mock.patch.object(core.memory, "load_recent_messages", return_value=[]),
                mock.patch.object(core.memory, "match_lesson_records", return_value=[]),
                mock.patch.object(core.memory, "save_message"),
                mock.patch.object(core.memory, "get_latest_plan", return_value=None),
                mock.patch.object(
                    core.memory,
                    "create_plan",
                    return_value=91,
                ) as create_plan,
                mock.patch.dict(
                    core.TOOL_IMPL,
                    {
                        "listening_ports": (
                            lambda **_kwargs:
                            TEST_LISTENING_PORTS_RESULT
                        ),
                    },
                    clear=False,
                ),
            ):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=True,
                    workdir=directory,
                    session_id=42,
                )

                reply = agent.step("Make a plan to create a local webpage.")

            self.assertEqual(client.chat.call_count, 4)
            self.assertIn("[Plan draft #91 is ready for approval.]", reply)
            create_plan.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
