import json
import os
import tempfile
import unittest
from unittest import mock

from agent import core


def fenced(payload):
    return "```liam-plan\n" + json.dumps(payload) + "\n```"


def plan_payload():
    return {
        "title": "Repair project",
        "objective": "Repair the observed project defects.",
        "files": [],
        "steps": [
            "Use the inspected project evidence to identify and repair the defect.",
        ],
        "validation": [
            {
                "command": "python3 -m unittest discover -s tests -t .",
                "expected": "The complete test suite exits successfully.",
            }
        ],
        "non_goals": [
            "Do not refactor unrelated code.",
        ],
        "risks": [
            "Additional defects may require further targeted inspection.",
        ],
    }


class MicroPlanningTests(unittest.TestCase):
    def build_agent(self, client, directory, create_plan):
        patches = [
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
                create_plan,
            ),
            mock.patch.dict(
                os.environ,
                {"LIAM_HELPER_OLLAMA_URL": ""},
                clear=False,
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        agent = core.Agent(
            channel="gui",
            plan_mode=True,
            workdir=directory,
            session_id=42,
        )
        agent.on_tool_call = mock.Mock()
        agent.on_status = mock.Mock()
        return agent

    def test_nested_micro_plan_is_rejected(self):
        first = core.Agent._start_micro_plan()

        with self.assertRaisesRegex(
            RuntimeError,
            "nested micro-plans are not allowed",
        ):
            core.Agent._start_micro_plan(first)

    def test_file_read_budget_forces_tool_free_synthesis(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(core.MICRO_PLAN_MAX_FILE_READS):
                path = os.path.join(directory, f"file{index}.txt")
                with open(path, "w") as handle:
                    handle.write(f"observed content {index}")
                paths.append(path)

            responses = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": path},
                        },
                    }],
                }
                for path in paths
            ]
            responses.append({
                "role": "assistant",
                "content": json.dumps(plan_payload()),
            })

            client = mock.Mock()
            client.chat.side_effect = responses
            create_plan = mock.Mock(return_value=91)
            agent = self.build_agent(
                client,
                directory,
                create_plan,
            )

            reply = agent.step(
                "Make a plan to fix the project in this folder."
            )

        self.assertEqual(
            client.chat.call_count,
            core.MICRO_PLAN_MAX_FILE_READS + 1,
        )
        final_call = client.chat.call_args_list[-1]
        self.assertEqual(final_call.kwargs["tools"], [])
        self.assertIs(
            final_call.kwargs["response_format"],
            core.PLAN_DRAFT_JSON_SCHEMA,
        )
        self.assertIn("```liam-plan", reply)
        self.assertIn(
            "[Plan draft #91 is ready for approval.]",
            reply,
        )
        create_plan.assert_called_once()

    def test_synthesis_tool_call_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(core.MICRO_PLAN_MAX_FILE_READS):
                path = os.path.join(directory, f"source{index}.txt")
                with open(path, "w") as handle:
                    handle.write(f"observed source {index}")
                paths.append(path)

            forbidden_path = os.path.join(
                directory,
                "must-not-be-read.txt",
            )
            with open(forbidden_path, "w") as handle:
                handle.write("unexpected nested discovery")

            responses = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": path},
                        },
                    }],
                }
                for path in paths
            ]
            responses.extend([
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "read_file",
                            "arguments": {
                                "path": forbidden_path,
                            },
                        },
                    }],
                },
                {
                    "role": "assistant",
                    "content": json.dumps(plan_payload()),
                },
            ])

            client = mock.Mock()
            client.chat.side_effect = responses
            create_plan = mock.Mock(return_value=93)
            agent = self.build_agent(
                client,
                directory,
                create_plan,
            )
            execute_tool = agent._execute_tool
            agent._execute_tool = mock.Mock(
                side_effect=execute_tool,
            )

            reply = agent.step(
                "Make a plan to fix the project in this folder."
            )

        self.assertEqual(
            client.chat.call_count,
            core.MICRO_PLAN_MAX_FILE_READS + 2,
        )
        self.assertNotIn(
            mock.call(
                "read_file",
                {"path": forbidden_path},
            ),
            agent._execute_tool.call_args_list,
        )
        self.assertEqual(
            client.chat.call_args_list[-2].kwargs["tools"],
            [],
        )
        self.assertEqual(
            client.chat.call_args_list[-1].kwargs["tools"],
            [],
        )
        self.assertIn("```liam-plan", reply)
        self.assertIn(
            "[Plan draft #93 is ready for approval.]",
            reply,
        )
        create_plan.assert_called_once()

    def test_forced_synthesis_removes_only_extra_placeholder_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(core.MICRO_PLAN_MAX_FILE_READS):
                path = os.path.join(directory, f"observed{index}.txt")
                with open(path, "w") as handle:
                    handle.write(f"observed project content {index}")
                paths.append(path)

            payload = plan_payload()
            payload["validation"].append({
                "command": "test -f TODO",
                "expected": "The additional project file exists.",
            })

            responses = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": path},
                        },
                    }],
                }
                for path in paths
            ]
            responses.append({
                "role": "assistant",
                "content": json.dumps(payload),
            })

            client = mock.Mock()
            client.chat.side_effect = responses
            create_plan = mock.Mock(return_value=94)
            agent = self.build_agent(
                client,
                directory,
                create_plan,
            )

            reply = agent.step(
                "Make a plan to fix the project in this folder."
            )

        self.assertEqual(
            client.chat.call_count,
            core.MICRO_PLAN_MAX_FILE_READS + 1,
        )
        self.assertIn("```liam-plan", reply)
        self.assertNotIn("TODO", reply)
        self.assertIn(
            "[Plan draft #94 is ready for approval.]",
            reply,
        )
        create_plan.assert_called_once()
        stored_call = repr(create_plan.call_args)
        self.assertIn(
            "python3 -m unittest discover -s tests -t .",
            stored_call,
        )
        self.assertNotIn("TODO", stored_call)

    def test_placeholder_is_not_silently_removed_before_synthesis(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid_payload = plan_payload()
            invalid_payload["validation"].append({
                "command": "test -f TODO",
                "expected": "The additional project file exists.",
            })

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": json.dumps(invalid_payload),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(plan_payload()),
                },
            ]
            create_plan = mock.Mock(return_value=95)
            agent = self.build_agent(
                client,
                directory,
                create_plan,
            )

            reply = agent.step(
                "Make a plan to fix the project in this folder."
            )

        self.assertEqual(client.chat.call_count, 2)
        self.assertIn(
            "[Plan draft #95 is ready for approval.]",
            reply,
        )
        create_plan.assert_called_once()

    def test_step_limit_gets_one_final_tool_free_synthesis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "README.md")
            with open(path, "w") as handle:
                handle.write("observed project")

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": path},
                        },
                    }],
                },
                {
                    "role": "assistant",
                    "content": json.dumps(plan_payload()),
                },
            ]
            create_plan = mock.Mock(return_value=92)
            agent = self.build_agent(
                client,
                directory,
                create_plan,
            )

            with (
                mock.patch.object(core, "MAX_STEPS", 1),
                mock.patch.object(
                    core,
                    "MICRO_PLAN_MAX_DISCOVERY_CALLS",
                    99,
                ),
                mock.patch.object(
                    core,
                    "MICRO_PLAN_MAX_FILE_READS",
                    99,
                ),
            ):
                reply = agent.step(
                    "Make a plan to fix the project in this folder."
                )

        self.assertEqual(client.chat.call_count, 2)
        final_call = client.chat.call_args_list[-1]
        self.assertEqual(final_call.kwargs["tools"], [])
        self.assertIs(
            final_call.kwargs["response_format"],
            core.PLAN_DRAFT_JSON_SCHEMA,
        )
        self.assertIn("```liam-plan", reply)
        self.assertIn(
            "[Plan draft #92 is ready for approval.]",
            reply,
        )
        create_plan.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
