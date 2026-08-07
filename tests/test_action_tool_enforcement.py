import unittest
from unittest import mock

from agent import core


class ActionToolRequirementTests(unittest.TestCase):
    def test_approved_plan_response_requires_real_tool(self):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = False
        agent._turn_plan_mode = False
        agent._active_plan_execution = {
            "plan_id": 41,
            "payload": {},
            "phase": "implementation",
            "step_number": 0,
            "current_step": "Modify index.html.",
        }

        required = agent._response_requires_real_tool(
            "ordinary host-generated execution prompt",
            "I updated the file.",
        )

        self.assertTrue(required)

    def test_approved_plan_repair_requires_real_tool(self):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = False
        agent._turn_plan_mode = False
        agent._active_plan_execution = {
            "plan_id": 41,
            "payload": {},
            "phase": "repair",
            "step_number": None,
            "current_step": None,
        }

        required = agent._response_requires_real_tool(
            "ordinary repair prompt",
            "I fixed the validation problem.",
        )

        self.assertTrue(required)

    def test_approved_plan_prefix_without_host_context_is_not_authoritative(self):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = False
        agent._turn_plan_mode = False
        agent._active_plan_execution = None

        required = agent._response_requires_real_tool(
            "[APPROVED PLAN EXECUTION]\nModify index.html.",
            "Informational response.",
        )

        self.assertFalse(required)

    def test_validation_phase_does_not_create_model_tool_requirement(self):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = False
        agent._turn_plan_mode = False
        agent._active_plan_execution = {
            "plan_id": 41,
            "payload": {},
            "phase": "validation",
            "step_number": None,
            "current_step": None,
        }

        required = agent._response_requires_real_tool(
            "validation bookkeeping",
            "Validation bookkeeping.",
        )

        self.assertFalse(required)

    def test_fake_edit_syntax_requires_real_tool(self):
        agent = core.Agent.__new__(core.Agent)

        required = agent._response_requires_real_tool(
            "Change the copyright year in index.html to 2026.",
            "```python\nedit_file(path='index.html')\n```",
        )

        self.assertTrue(required)

    def test_use_ogg_to_play_is_an_action_request(self):
        agent = core.Agent.__new__(core.Agent)

        self.assertTrue(
            agent._plan_required_for_request(
                "Can you use ogg to play this file?"
            )
        )

    def test_explicit_run_shell_request_selects_exact_tool(self):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = False
        agent.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "description": "Run a shell command.",
                    "parameters": {},
                },
            },
        ]

        selected = agent._explicit_requested_tool_schemas(
            (
                "Look back at the run_shell_command you used and "
                "do that with this file."
            ),
            agent.tool_schemas,
        )

        self.assertEqual(
            [
                schema["function"]["name"]
                for schema in selected
            ],
            ["run_shell_command"],
        )

    def test_action_promise_requires_real_tool(self):
        agent = core.Agent.__new__(core.Agent)

        required = agent._response_requires_real_tool(
            "Run pkill ogg123.",
            "I will now execute the command.",
        )

        self.assertTrue(required)

    def test_terse_action_followups_require_tool_for_promise(self):
        agent = core.Agent.__new__(core.Agent)

        requests = (
            "do it",
            "run it",
            "well run the pkill command you just described",
            "fucking run the pkill command",
            'do this: run_shell_command({"command": "pkill ogg123"})',
            "try again and run it",
        )

        for user_input in requests:
            with self.subTest(user_input=user_input):
                required = agent._response_requires_real_tool(
                    user_input,
                    "I will use the run_shell_command tool now.",
                )
                self.assertTrue(required)

    def test_action_followup_can_report_concrete_blocker(self):
        agent = core.Agent.__new__(core.Agent)

        required = agent._response_requires_real_tool(
            "do it",
            "I cannot identify which command you mean.",
        )

        self.assertFalse(required)

    def test_informational_answer_does_not_require_tool(self):
        agent = core.Agent.__new__(core.Agent)

        required = agent._response_requires_real_tool(
            "What year is shown in index.html?",
            "The file shows 2023.",
        )

        self.assertFalse(required)

    def test_plan_mode_draft_does_not_require_execution_tool(self):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = True

        required = agent._response_requires_real_tool(
            "Make a plan to create a local webpage.",
            "```liam-plan\n{\"title\": \"Create webpage\"}\n```",
        )

        self.assertFalse(required)

    def test_pending_action_survives_non_action_followup(self):
        followups = (
            "Did you run that command?",
            "bullshit",
        )

        for followup in followups:
            with self.subTest(followup=followup):
                agent = core.Agent.__new__(core.Agent)
                agent.plan_mode = False
                agent.messages = [
                    {
                        "role": "user",
                        "content": (
                            "Play /var/www/AMusic/song.flac using ogg123."
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "I will run the command now.",
                    },
                    {
                        "role": "user",
                        "content": followup,
                    },
                ]

                required = agent._response_requires_real_tool(
                    followup,
                    "I will use run_shell_command to execute it now.",
                )

                self.assertTrue(required)

    def test_pending_action_allows_honest_nonexecution_report(self):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = False
        agent.messages = [
            {
                "role": "user",
                "content": "Run printf action-ran.",
            },
            {
                "role": "assistant",
                "content": "I will run the command now.",
            },
            {
                "role": "user",
                "content": "Did you run that command?",
            },
        ]

        required = agent._response_requires_real_tool(
            "Did you run that command?",
            "No. I did not run the command.",
        )

        self.assertFalse(required)

    @mock.patch.object(core.memory, "load_recent_notes", return_value=[])
    @mock.patch.object(core.memory, "load_recent_messages", return_value=[])
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "get_latest_plan", return_value=None)
    def test_pending_action_followup_recovery_executes_real_tool(
        self,
        _get_latest_plan,
        _save_message,
        _match_lessons,
        _load_messages,
        _load_notes,
    ):
        import tempfile

        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": "I will use run_shell_command now.",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_shell_command",
                            "arguments": {
                                "command": "printf action-ran"
                            },
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "The command completed.",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(core, "OllamaClient", return_value=client):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=False,
                    workdir=directory,
                    session_id=53,
                )

            agent.messages = [
                {
                    "role": "user",
                    "content": "Run printf action-ran.",
                },
                {
                    "role": "assistant",
                    "content": "I will run the command now.",
                },
            ]

            recovery_schema = next(
                schema
                for schema in agent.tool_schemas
                if schema["function"]["name"] == "run_shell_command"
            )
            agent._select_recovery_tool_schemas = mock.Mock(
                return_value=[recovery_schema]
            )
            agent.on_tool_call = mock.Mock()

            def execute_tool(name, args):
                agent._tool_events.append({
                    "tool": name,
                    "status": "success",
                    "args": args,
                    "result": "action-ran",
                })
                return "action-ran"

            agent._execute_tool = mock.Mock(side_effect=execute_tool)

            result = agent.step("Did you run that command?")

        self.assertEqual(result, "The command completed.")
        agent._execute_tool.assert_called_once_with(
            "run_shell_command",
            {"command": "printf action-ran"},
        )
        self.assertEqual(client.chat.call_count, 3)

    @mock.patch.object(core.memory, "load_recent_notes", return_value=[])
    @mock.patch.object(core.memory, "load_recent_messages", return_value=[])
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "get_latest_plan", return_value=None)
    def test_wrong_read_tool_cannot_complete_play_action(
        self,
        _get_latest_plan,
        _save_message,
        _match_lessons,
        _load_messages,
        _load_notes,
    ):
        import tempfile

        path = "/media/example/Ride.flac"

        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": path},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": (
                    "The file contents do not show whether it can play."
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_shell_command",
                            "arguments": {
                                "command": f'ogg123 "{path}"'
                            },
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "Playback started.",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(core, "OllamaClient", return_value=client):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=False,
                    workdir=directory,
                    session_id=54,
                )

            run_schema = next(
                schema
                for schema in agent.tool_schemas
                if schema["function"]["name"]
                == "run_shell_command"
            )
            agent._select_recovery_tool_schemas = mock.Mock(
                return_value=[run_schema]
            )
            agent.on_tool_call = mock.Mock()

            def execute_tool(name, args):
                if name == "read_file":
                    result = "binary media data"
                elif name == "run_shell_command":
                    result = "Playing Ride.flac\n[exit code: 0]"
                else:
                    self.fail(f"unexpected tool: {name}")

                agent._tool_events.append({
                    "tool": name,
                    "status": "success",
                    "args": args,
                    "result": result,
                })
                return result

            agent._execute_tool = mock.Mock(side_effect=execute_tool)

            result = agent.step(
                f"Can you use ogg to play this file?\n{path}"
            )

        self.assertEqual(result, "Playback started.")
        self.assertEqual(
            [
                call.args[0]
                for call in agent._execute_tool.call_args_list
            ],
            ["read_file", "run_shell_command"],
        )
        self.assertEqual(client.chat.call_count, 4)

    @mock.patch.object(core.memory, "load_recent_notes", return_value=[])
    @mock.patch.object(core.memory, "load_recent_messages", return_value=[])
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "get_latest_plan", return_value=None)
    def test_action_promise_recovery_executes_real_tool(
        self,
        _get_latest_plan,
        _save_message,
        _match_lessons,
        _load_messages,
        _load_notes,
    ):
        import tempfile

        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": "I will run the command now.",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "run_shell_command",
                            "arguments": {"command": "printf action-ran"},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "The command completed.",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(core, "OllamaClient", return_value=client):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=False,
                    workdir=directory,
                    session_id=51,
                )

            recovery_schema = next(
                schema
                for schema in agent.tool_schemas
                if schema["function"]["name"] == "run_shell_command"
            )
            agent._select_recovery_tool_schemas = mock.Mock(
                return_value=[recovery_schema]
            )
            agent.on_tool_call = mock.Mock()

            def execute_tool(name, args):
                self.assertEqual(name, "run_shell_command")
                self.assertEqual(args, {"command": "printf action-ran"})
                agent._tool_events.append({
                    "tool": name,
                    "status": "success",
                    "args": args,
                    "result": "action-ran",
                })
                return "action-ran"

            agent._execute_tool = mock.Mock(side_effect=execute_tool)

            result = agent.step("Run printf action-ran.")

        self.assertEqual(result, "The command completed.")
        agent._execute_tool.assert_called_once_with(
            "run_shell_command",
            {"command": "printf action-ran"},
        )
        self.assertEqual(client.chat.call_count, 3)
        self.assertEqual(
            agent._tool_events,
            [
                {
                    "tool": "run_shell_command",
                    "status": "success",
                    "args": {"command": "printf action-ran"},
                    "result": "action-ran",
                }
            ],
        )

    @mock.patch.object(core.memory, "load_recent_notes", return_value=[])
    @mock.patch.object(core.memory, "load_recent_messages", return_value=[])
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "get_latest_plan", return_value=None)
    def test_repeated_action_promise_returns_host_failure(
        self,
        _get_latest_plan,
        _save_message,
        _match_lessons,
        _load_messages,
        _load_notes,
    ):
        import tempfile

        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": "I will run the command now.",
            },
            {
                "role": "assistant",
                "content": "I will use run_shell_command now.",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(core, "OllamaClient", return_value=client):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=False,
                    workdir=directory,
                    session_id=52,
                )

            recovery_schema = next(
                schema
                for schema in agent.tool_schemas
                if schema["function"]["name"] == "run_shell_command"
            )
            agent._select_recovery_tool_schemas = mock.Mock(
                return_value=[recovery_schema]
            )
            agent._execute_tool = mock.Mock()

            result = agent.step("Run printf action-ran.")

        self.assertIn(
            "Liam retried tool selection but still did not call a tool.",
            result,
        )
        self.assertIn("No action was performed.", result)
        agent._execute_tool.assert_not_called()
        self.assertEqual(client.chat.call_count, 2)

    @mock.patch.object(core.memory, "load_recent_notes", return_value=[])
    @mock.patch.object(core.memory, "load_recent_messages", return_value=[])
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "get_latest_plan", return_value=None)
    @mock.patch.object(core.memory, "create_plan", return_value=92)
    def test_explicit_plan_request_uses_temporary_plan_mode(
        self,
        create_plan,
        _get_latest_plan,
        _save_message,
        _match_lessons,
        _load_messages,
        _load_notes,
    ):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "settings.txt")
            with open(path, "w") as handle:
                handle.write("year=2023\n")

            payload = {
                "version": core.PLAN_VERSION,
                "title": "Update year",
                "objective": "Update the configured year to 2026.",
                "files": [path],
                "steps": [
                    f"Edit {path} so year is 2026.",
                ],
                "work_units": [
                    {
                        "description": f"Edit {path} so year is 2026.",
                        "tool": "edit_file",
                        "arguments": {
                            "path": path,
                            "old_string": "year=2023\n",
                            "new_string": "year=2026\n",
                        },
                    }
                ],
                "validation": [
                    {
                        "command": f"grep -q '^year=2026$' {path}",
                        "expected": "The file contains year=2026.",
                    }
                ],
                "non_goals": [
                    f"Do not modify files outside of {directory}."
                ],
                "risks": ["The expected original line may be absent."],
            }

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": path},
                            }
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": (
                        "```liam-plan\n"
                        + json.dumps(payload)
                        + "\n```"
                    ),
                },
                {
                    "role": "assistant",
                    "content": "Four.",
                },
            ]

            with mock.patch.object(
                core,
                "OllamaClient",
                return_value=client,
            ):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=False,
                    workdir=directory,
                    session_id=42,
                )

                reply = agent.step(
                    "Make a new plan to examine settings.txt and "
                    "update the year to 2026."
                )

                self.assertIn(
                    "[Plan draft #92 is ready for approval.]",
                    reply,
                )
                self.assertFalse(agent.plan_mode)
                create_plan.assert_called_once()

                first_tools = {
                    schema["function"]["name"]
                    for schema in client.chat.call_args_list[0].kwargs["tools"]
                }
                self.assertIn("read_file", first_tools)
                self.assertNotIn("edit_file", first_tools)
                self.assertTrue(
                    first_tools.issubset(core.PLAN_MODE_ALLOWED_TOOLS)
                )

                ordinary_reply = agent.step("What is two plus two?")

                self.assertEqual(ordinary_reply, "Four.")
                self.assertFalse(agent._turn_plan_mode)
                third_tools = {
                    schema["function"]["name"]
                    for schema in client.chat.call_args_list[2].kwargs["tools"]
                }
                self.assertIn("edit_file", third_tools)

    def test_explicit_plan_request_variants_are_narrowly_recognized(self):
        self.assertTrue(
            core.Agent._is_explicit_plan_request(
                "Make a plan to update settings.txt."
            )
        )
        self.assertTrue(
            core.Agent._is_explicit_plan_request(
                "Make a new plan to update settings.txt."
            )
        )
        self.assertTrue(
            core.Agent._is_explicit_plan_request(
                "Create a new implementation plan for the website."
            )
        )
        self.assertFalse(
            core.Agent._is_explicit_plan_request(
                "Explain the new plan to me."
            )
        )
        self.assertFalse(
            core.Agent._is_explicit_plan_request(
                "We may need a new plan later."
            )
        )

    def test_directory_read_automatically_uses_list_directory(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            expected = os.path.join(directory, "style.css")
            with open(expected, "w") as handle:
                handle.write("body {}\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent.plan_mode = False
            agent._turn_plan_mode = False
            agent.channel = "gui"
            agent.allowed_tools = None
            agent.notes_session_id = None
            agent.session_id = None
            agent._current_user_input = ""
            agent._read_paths_this_turn = set()
            agent._tool_events = []
            agent.auto_confirm = True
            agent.on_confirm = lambda _name, _args: True
            agent.on_tool_call = mock.Mock()

            result = agent._execute_tool(
                "read_file",
                {"path": directory},
            )

            self.assertIn("automatically used list_directory", result)
            self.assertIn("style.css", result)
            agent.on_tool_call.assert_called_once_with(
                "list_directory",
                {"path": directory},
            )
            self.assertEqual(
                [event["status"] for event in agent._tool_events],
                ["failure", "success"],
            )
            self.assertEqual(
                [event["tool"] for event in agent._tool_events],
                ["read_file", "list_directory"],
            )

    def test_plan_rejects_declared_file_without_step_reference(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")

            with open(index_path, "w") as handle:
                handle.write("<html></html>\n")
            with open(style_path, "w") as handle:
                handle.write("body {}\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {
                index_path,
                style_path,
            }

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [
                    index_path,
                    style_path,
                ],
                "steps": [
                    "Update the copyright year in index.html.",
                ],
            }))

            self.assertIsNotNone(problem)
            self.assertIn("no implementation step reference", problem)
            self.assertIn("style.css", problem)

    def test_declared_filename_does_not_match_longer_extension(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")

            with open(index_path, "w") as handle:
                handle.write("<html></html>\n")
            with open(style_path, "w") as handle:
                handle.write("body {}\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {
                index_path,
                style_path,
            }

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [
                    index_path,
                    style_path,
                ],
                "steps": [
                    "Update index.html.bak but do not change the originals.",
                ],
            }))

            self.assertIsNone(problem)

    def test_plan_generic_steps_without_named_files_remain_allowed(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "first.txt")
            second = os.path.join(directory, "second.txt")

            for path in (first, second):
                with open(path, "w") as handle:
                    handle.write("content\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {
                first,
                second,
            }

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [first, second],
                "steps": [
                    "Apply the minimum required code changes.",
                ],
            }))

            self.assertIsNone(problem)

    def test_plan_rejects_unaddressed_missing_html_dependency(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            with open(index_path, "w") as handle:
                handle.write(
                    '<html><script src="script.js"></script></html>\n'
                )

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {index_path}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [index_path],
                "steps": [
                    "Update the copyright year in index.html.",
                ],
            }))

            self.assertIsNotNone(problem)
            self.assertIn("missing local dependency", problem)
            self.assertIn("script.js", problem)

    def test_plan_allows_missing_html_dependency_when_addressed(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            with open(index_path, "w") as handle:
                handle.write(
                    '<html><script src="script.js"></script></html>\n'
                )

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {index_path}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [index_path],
                "steps": [
                    "Update the copyright year in index.html.",
                    (
                        "Remove the broken script.js reference from "
                        "index.html."
                    ),
                ],
            }))

            self.assertIsNone(problem)

    def test_interactive_javascript_requires_integration_validation(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")
            script_path = os.path.join(directory, "script.js")

            with open(index_path, "w") as handle:
                handle.write(
                    '<button id="theme-toggle">Toggle</button>'
                    '<script src="script.js"></script>\n'
                )
            with open(style_path, "w") as handle:
                handle.write(".dark-mode { color: white; }\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {
                index_path,
                style_path,
            }

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [
                    index_path,
                    style_path,
                    script_path,
                ],
                "steps": [
                    "Reference index.html while preserving its existing button.",
                    "Reference style.css while preserving its theme rules.",
                    (
                        "Create script.js and implement JavaScript to toggle "
                        "the theme when the button is clicked."
                    ),
                ],
                "validation": [
                    {
                        "command": (
                            "grep -q 'toggleDarkMode()' "
                            + script_path
                        ),
                        "expected": "The function exists.",
                    },
                ],
            }))

            self.assertIsNotNone(problem)
            self.assertIn(
                "interactive JavaScript validation",
                problem,
            )
            self.assertIn(
                "HTML control identifier",
                problem,
            )
            self.assertIn(
                "event-binding mechanism",
                problem,
            )
            self.assertIn(
                "CSS state class",
                problem,
            )

    def test_standalone_javascript_creation_needs_no_integration_evidence(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            script_path = os.path.join(directory, "script.js")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = set()

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [script_path],
                "steps": [
                    "Create script.js as a new file for a theme toggle.",
                ],
                "validation": [],
            }))

            self.assertIsNone(problem)

    def test_interactive_javascript_uses_undeclared_inspected_css_evidence(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")
            script_path = os.path.join(directory, "script.js")

            with open(index_path, "w") as handle:
                handle.write(
                    '<button id="theme-toggle">Toggle</button>'
                    '<script src="script.js"></script>\n'
                )
            with open(style_path, "w") as handle:
                handle.write(".dark-mode { color: white; }\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {
                index_path,
                style_path,
            }

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [
                    index_path,
                    script_path,
                ],
                "steps": [
                    "Update index.html while preserving its existing button.",
                    (
                        "Create script.js and implement the theme toggle "
                        "when the button is clicked."
                    ),
                ],
                "validation": [
                    {
                        "command": (
                            "grep -Fq 'toggleDarkMode()' "
                            + script_path
                        ),
                        "expected": "The theme function exists.",
                    },
                ],
            }))

            self.assertIsNotNone(problem)
            self.assertIn("'theme-toggle'", problem)
            self.assertIn("addEventListener", problem)
            self.assertIn("'dark-mode'", problem)

    def test_interactive_javascript_reports_only_relevant_evidence(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")
            script_path = os.path.join(directory, "script.js")

            with open(index_path, "w") as handle:
                handle.write(
                    '<button id="theme-toggle">Toggle</button>'
                    '<section id="movies"></section>'
                    '<section id="music"></section>'
                    '<section id="tv-shows"></section>'
                    '<script src="script.js"></script>\n'
                )
            with open(style_path, "w") as handle:
                handle.write(
                    ".dark-mode { color: white; }\n"
                    ".light-mode { color: black; }\n"
                )

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {
                index_path,
                style_path,
            }

            problem = agent._plan_file_evidence_problem(json.dumps({
                "objective": (
                    "Implement a light and dark theme toggle."
                ),
                "files": [
                    index_path,
                    script_path,
                ],
                "steps": [
                    "Update index.html while preserving theme-toggle.",
                    (
                        "Create script.js and implement the light and dark "
                        "theme toggle when the button is clicked."
                    ),
                ],
                "validation": [
                    {
                        "command": (
                            "grep -Fq 'toggleDarkMode()' "
                            + script_path
                        ),
                        "expected": "The function exists.",
                    },
                ],
            }))

            self.assertIsNotNone(problem)
            self.assertIn("'theme-toggle'", problem)
            self.assertIn("'dark-mode'", problem)
            self.assertIn("'light-mode'", problem)
            self.assertNotIn("'movies'", problem)
            self.assertNotIn("'music'", problem)
            self.assertNotIn("'tv-shows'", problem)

    def test_expected_text_cannot_replace_executable_integration_checks(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")
            script_path = os.path.join(directory, "script.js")

            with open(index_path, "w") as handle:
                handle.write(
                    '<button id="theme-toggle">Toggle</button>'
                    '<script src="script.js"></script>\n'
                )
            with open(style_path, "w") as handle:
                handle.write(".dark-mode { color: white; }\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {
                index_path,
                style_path,
            }

            problem = agent._plan_file_evidence_problem(json.dumps({
                "objective": "Implement a dark theme toggle.",
                "files": [
                    index_path,
                    script_path,
                ],
                "steps": [
                    "Update index.html while preserving theme-toggle.",
                    (
                        "Create script.js and implement the dark theme "
                        "toggle when the button is clicked."
                    ),
                ],
                "validation": [
                    {
                        "command": (
                            "grep -Fq 'toggleDarkMode()' "
                            + script_path
                        ),
                        "expected": (
                            "theme-toggle uses addEventListener and "
                            "changes dark-mode."
                        ),
                    },
                ],
            }))

            self.assertIsNotNone(problem)
            self.assertIn(
                "interactive JavaScript validation",
                problem,
            )

    def test_interactive_javascript_integration_validation_is_allowed(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")
            script_path = os.path.join(directory, "script.js")

            with open(index_path, "w") as handle:
                handle.write(
                    '<button id="theme-toggle">Toggle</button>'
                    '<script src="script.js"></script>\n'
                )
            with open(style_path, "w") as handle:
                handle.write(".dark-mode { color: white; }\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {
                index_path,
                style_path,
            }

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [
                    index_path,
                    style_path,
                    script_path,
                ],
                "steps": [
                    "Reference index.html while preserving its existing button.",
                    "Reference style.css while preserving its theme rules.",
                    (
                        "Create script.js and implement JavaScript to toggle "
                        "the theme when the button is clicked."
                    ),
                ],
                "validation": [
                    {
                        "command": (
                            "grep -Fq 'theme-toggle' "
                            + script_path
                            + " && grep -Fq 'addEventListener' "
                            + script_path
                            + " && grep -Fq 'dark-mode' "
                            + script_path
                        ),
                        "expected": (
                            "script.js binds theme-toggle with "
                            "addEventListener and toggles dark-mode."
                        ),
                    },
                    {
                        "command": (
                            "grep -Fq '.dark-mode {' "
                            + style_path
                        ),
                        "expected": (
                            "style.css contains the exact inspected "
                            "dark-mode selector."
                        ),
                    },
                ],
            }))

            self.assertIsNone(problem)

    def test_validation_rejects_false_assertion_on_unchanged_file(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, "style.css")
            with open(style_path, "w") as handle:
                handle.write(".dark-mode { color: white; }\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {style_path}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [],
                "steps": [
                    "Implement the required JavaScript behavior.",
                ],
                "validation": [
                    {
                        "command": (
                            "grep -Fq '.dark-mode, .light-mode {' "
                            + style_path
                        ),
                        "expected": "Both theme selectors are defined.",
                    },
                ],
            }))

            self.assertIsNotNone(problem)
            self.assertIn(
                "asserts missing content",
                problem,
            )
            self.assertIn("unchanged local file", problem)

    def test_validation_allows_true_assertion_on_unchanged_file(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, "style.css")
            with open(style_path, "w") as handle:
                handle.write(".dark-mode { color: white; }\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {style_path}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [],
                "steps": [
                    "Implement the required JavaScript behavior.",
                ],
                "validation": [
                    {
                        "command": (
                            "grep -Fq '.dark-mode {' "
                            + style_path
                        ),
                        "expected": "The existing dark-mode selector remains.",
                    },
                ],
            }))

            self.assertIsNone(problem)

    def test_validation_does_not_preflight_declared_change_target(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, "style.css")
            with open(style_path, "w") as handle:
                handle.write(".dark-mode { color: white; }\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {style_path}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [style_path],
                "steps": [
                    "Modify style.css to add a combined theme selector.",
                ],
                "validation": [
                    {
                        "command": (
                            "grep -Fq '.dark-mode, .light-mode {' "
                            + style_path
                        ),
                        "expected": "The new combined selector exists.",
                    },
                ],
            }))

            self.assertIsNone(problem)

    def test_plan_rejects_adding_existing_css_classes(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, "style.css")
            with open(style_path, "w") as handle:
                handle.write(
                    ".dark-mode { color: white; }\n"
                    ".light-mode { color: black; }\n"
                )

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {style_path}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [style_path],
                "steps": [
                    (
                        "Add CSS classes for dark and light modes "
                        "in style.css."
                    ),
                ],
                "validation": [],
            }))

            self.assertIsNotNone(problem)
            self.assertIn(
                "already present in inspected file",
                problem,
            )
            self.assertIn("'dark-mode'", problem)
            self.assertIn("'light-mode'", problem)

    def test_plan_allows_modifying_existing_css_classes(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, "style.css")
            with open(style_path, "w") as handle:
                handle.write(
                    ".dark-mode { color: white; }\n"
                    ".light-mode { color: black; }\n"
                )

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {style_path}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [style_path],
                "steps": [
                    (
                        "Modify the existing dark-mode and light-mode "
                        "classes in style.css to add a new property."
                    ),
                ],
                "validation": [],
            }))

            self.assertIsNone(problem)

    def test_plan_allows_adding_missing_css_class(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, "style.css")
            with open(style_path, "w") as handle:
                handle.write(".dark-mode { color: white; }\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {style_path}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [style_path],
                "steps": [
                    "Add the CSS class contrast-mode to style.css.",
                ],
                "validation": [],
            }))

            self.assertIsNone(problem)

    def test_unchanged_css_error_includes_exact_transition_candidates(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, "style.css")
            with open(style_path, "w") as handle:
                handle.write(
                    ":root { --transition-speed: 0.3s; }\n"
                    "body { transition: background-color "
                    "var(--transition-speed), color "
                    "var(--transition-speed); }\n"
                    "header { transition: background-color "
                    "var(--transition-speed); }\n"
                )

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {style_path}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [],
                "steps": ["Use the existing stylesheet behavior."],
                "validation": [{
                    "command": (
                        "grep -Fq 'transition: all 0.5s ease;' "
                        + style_path
                    ),
                    "expected": "Theme transitions exist.",
                }],
            }))

            self.assertIsNotNone(problem)
            self.assertIn(
                "exact inspected candidate literals include",
                problem,
            )
            self.assertIn(
                "transition: background-color "
                "var(--transition-speed), color "
                "var(--transition-speed);",
                problem,
            )
            self.assertNotIn(
                "'transition: all 0.5s ease;' |",
                problem,
            )

    def test_unchanged_css_error_includes_exact_selector_candidates(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, "style.css")
            with open(style_path, "w") as handle:
                handle.write(
                    ".dark-mode { color: white; }\n"
                    ".light-mode { color: black; }\n"
                )

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {style_path}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [],
                "steps": ["Reuse the existing theme selectors."],
                "validation": [{
                    "command": (
                        "grep -Fq '.dark-mode, .light-mode {' "
                        + style_path
                    ),
                    "expected": "Theme selectors exist.",
                }],
            }))

            self.assertIsNotNone(problem)
            self.assertIn(
                "'.dark-mode { color: white; }'",
                problem,
            )
            self.assertIn(
                "'.light-mode { color: black; }'",
                problem,
            )

    def test_plan_rejects_existing_but_ungrounded_outside_path(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as workdir, \
             tempfile.TemporaryDirectory() as outside:
            target = os.path.join(outside, "project.txt")
            with open(target, "w") as handle:
                handle.write("real but unrelated\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = workdir
            agent.extra_folders = []
            agent._current_user_input = (
                "Make a plan to fix the project in this folder."
            )
            agent._read_paths_this_turn = {target}
            agent._tool_events = []

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [target],
                "steps": [f"Modify {target}."],
            }))

        self.assertIsNotNone(problem)
        self.assertIn("ungrounded path", problem)

    def test_plan_allows_explicit_user_named_outside_path_after_read(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as workdir, \
             tempfile.TemporaryDirectory() as outside:
            target = os.path.join(outside, "shared.txt")
            with open(target, "w") as handle:
                handle.write("shared\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = workdir
            agent.extra_folders = []
            agent._current_user_input = (
                f"Update the existing file {target}."
            )
            agent._read_paths_this_turn = {target}
            agent._tool_events = []

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [target],
                "steps": [f"Modify {target}."],
            }))

        self.assertIsNone(problem)

    def test_plan_allows_outside_path_grounded_by_project_evidence(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as workdir, \
             tempfile.TemporaryDirectory() as outside:
            config = os.path.join(workdir, "config.txt")
            target = os.path.join(outside, "shared.txt")

            with open(config, "w") as handle:
                handle.write(f"shared_path={target}\n")
            with open(target, "w") as handle:
                handle.write("shared\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = workdir
            agent.extra_folders = []
            agent._current_user_input = (
                "Make a plan to fix the project in this folder."
            )
            agent._read_paths_this_turn = {config, target}
            agent._tool_events = [
                {
                    "tool": "read_file",
                    "args": {"path": config},
                    "result": f"shared_path={target}\n",
                    "status": "success",
                },
                {
                    "tool": "read_file",
                    "args": {"path": target},
                    "result": "shared\n",
                    "status": "success",
                },
            ]

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [target],
                "steps": [f"Modify {target}."],
            }))

        self.assertIsNone(problem)

    def test_plan_rejects_ungrounded_absolute_validation_path(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            target = os.path.join(workdir, "config.txt")
            with open(target, "w") as handle:
                handle.write("enabled=true\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = workdir
            agent.extra_folders = []
            agent._current_user_input = (
                "Make a plan to update config.txt in this project."
            )
            agent._read_paths_this_turn = {target}
            agent._tool_events = []

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [target],
                "steps": [f"Modify {target}."],
                "validation": [{
                    "command": (
                        "grep -Fq enabled "
                        "/home/jupyter/project"
                    ),
                    "expected": "The configuration is enabled.",
                }],
            }))

        self.assertIsNotNone(problem)
        self.assertIn(
            "references ungrounded absolute path "
            "'/home/jupyter/project'",
            problem,
        )

    def test_validation_absolute_executable_is_not_treated_as_project_path(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as workdir:
            target = os.path.join(workdir, "config.txt")
            with open(target, "w") as handle:
                handle.write("enabled=true\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = workdir
            agent.extra_folders = []
            agent._current_user_input = (
                "Make a plan to update config.txt in this project."
            )
            agent._read_paths_this_turn = {target}
            agent._tool_events = []

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": [target],
                "steps": [f"Modify {target}."],
                "validation": [{
                    "command": (
                        f"/usr/bin/grep -Fq enabled {target}"
                    ),
                    "expected": "The configuration is enabled.",
                }],
            }))

        self.assertIsNone(problem)

    def test_approved_execution_rejects_existing_unrelated_path(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as workdir, \
             tempfile.TemporaryDirectory() as outside:
            target = os.path.join(outside, "unrelated.txt")
            with open(target, "w") as handle:
                handle.write("real but unrelated\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = workdir
            agent.extra_folders = []
            agent._current_user_input = ""
            agent._tool_events = []
            agent._active_plan_execution = {
                "plan_id": 41,
                "payload": {
                    "files": [],
                    "steps": [
                        "Fix the project in the thread folder."
                    ],
                    "validation": [],
                },
                "phase": "implementation",
                "step_number": 0,
                "current_step": (
                    "Fix the project in the thread folder."
                ),
            }

            problem = agent._approved_plan_path_problem(
                "file_info",
                {"path": target},
            )

        self.assertIsNotNone(problem)
        self.assertIn("rejected ungrounded", problem)
        self.assertIn(
            "existence on disk does not make it relevant",
            problem,
        )

    def test_plan_rejects_existing_file_not_read_this_turn(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            existing = os.path.join(directory, "index.html")
            with open(existing, "w") as handle:
                handle.write("<html></html>\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = set()

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": ["index.html"],
                "steps": [
                    "Edit index.html to update the copyright year.",
                ],
            }))

            self.assertIsNotNone(problem)
            self.assertIn("index.html", problem)
            self.assertIn("not inspected with read_file", problem)

    def test_plan_allows_existing_file_read_this_turn(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            existing = os.path.join(directory, "index.html")
            with open(existing, "w") as handle:
                handle.write("<html></html>\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {existing}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": ["index.html"],
                "steps": [
                    "Edit index.html to update the copyright year.",
                ],
            }))

            self.assertIsNone(problem)

    def test_plan_rejects_probable_typo_from_existing_sibling(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            actual = os.path.join(directory, "style.css")
            with open(actual, "w") as handle:
                handle.write("body {}\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = set()

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": ["styles.css"],
                "steps": [
                    "Add light and dark theme rules to styles.css.",
                ],
            }))

            self.assertIsNotNone(problem)
            self.assertIn("styles.css", problem)
            self.assertIn("style.css", problem)
            self.assertIn("intended target", problem)

    def test_plan_rejects_probable_typo_for_inspected_file(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            actual = os.path.join(directory, "style.css")
            with open(actual, "w") as handle:
                handle.write("body {}\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {actual}

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": ["styles.css"],
                "steps": [
                    "Add light and dark theme rules to styles.css.",
                ],
            }))

            self.assertIsNotNone(problem)
            self.assertIn("styles.css", problem)
            self.assertIn("style.css", problem)
            self.assertIn("intended target", problem)

    def test_plan_allows_explicit_creation_of_new_local_file(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = set()

            problem = agent._plan_file_evidence_problem(json.dumps({
                "files": ["script.js"],
                "steps": [
                    "Create script.js as a new file for the theme toggle.",
                ],
            }))

            self.assertIsNone(problem)

    def test_successful_tool_event_is_authoritative(self):
        self.assertTrue(core.Agent._has_successful_tool_event([
            {"tool": "read_file", "status": "success"},
        ]))
        self.assertFalse(core.Agent._has_successful_tool_event([
            {"tool": "edit_file", "status": "failure"},
            {"tool": "read_file", "status": "noop"},
        ]))

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_plan_work_unit_without_matching_event_fails_without_ai_retry(
        self,
        get_plan,
        _transition_plan,
    ):
        import json

        payload = {
            "version": core.PLAN_VERSION,
            "title": "Run approved command",
            "objective": "Run the exact approved command.",
            "files": [],
            "steps": ["Run the approved command."],
            "work_units": [
                {
                    "description": "Run the approved command.",
                    "tool": "run_shell_command",
                    "arguments": {
                        "command": "printf test",
                    },
                    "affected_paths": [],
                }
            ],
            "validation": [
                {
                    "command": "true",
                    "expected": "Exit code 0.",
                }
            ],
            "non_goals": ["Do not perform unrelated work."],
            "risks": ["The command may fail."],
        }
        get_plan.return_value = {
            "id": 9,
            "session_id": 17,
            "status": "approved",
            "content": json.dumps(payload),
        }

        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = False
        agent.session_id = 17
        agent.on_status = mock.Mock()
        agent.on_tool_call = mock.Mock()
        agent._tool_events = []
        agent.step = mock.Mock()
        agent._execute_tool = mock.Mock(return_value="command completed")
        agent._run_plan_validation = mock.Mock()
        agent._fail_running_plan = mock.Mock(
            return_value="FAIL: no successful tool event"
        )

        result = agent.execute_plan(9)

        self.assertEqual(result, "FAIL: no successful tool event")
        agent.step.assert_not_called()
        agent._execute_tool.assert_called_once_with(
            "run_shell_command",
            {
                "command": "printf test",
            },
        )
        agent._run_plan_validation.assert_not_called()
        agent._fail_running_plan.assert_called_once()
        self.assertIn(
            "did not produce the exact host-observed completion event",
            agent._fail_running_plan.call_args.args[1],
        )

    def test_reuse_step_with_verified_fetch_needs_no_listening_ports(self):
        import json

        agent = core.Agent.__new__(core.Agent)
        agent.workdir = "/var/www/LiamApp01"
        agent._read_paths_this_turn = set()
        agent.plan_mode = True
        agent._tool_events = [
            {
                "tool": "fetch_url",
                "status": "success",
                "args": {"url": "http://127.0.0.1:8002/"},
            },
        ]

        problem = agent._plan_file_evidence_problem(json.dumps({
            "objective": (
                "Restyle the local webpage already served at "
                "http://127.0.0.1:8002."
            ),
            "files": [],
            "steps": [
                (
                    "Reuse the already-running server on port 8002; "
                    "no new server is needed."
                ),
            ],
            "validation": [],
        }))

        self.assertIsNone(problem)

    def test_reuse_step_without_verified_fetch_still_requires_listening_ports(self):
        import json

        agent = core.Agent.__new__(core.Agent)
        agent.workdir = "/var/www/LiamApp01"
        agent._read_paths_this_turn = set()
        agent.plan_mode = True
        agent._tool_events = []

        problem = agent._plan_file_evidence_problem(json.dumps({
            "objective": (
                "Restyle the local webpage already served at "
                "http://127.0.0.1:8002."
            ),
            "files": [],
            "steps": [
                (
                    "Reuse the already-running server on port 8002; "
                    "no new server is needed."
                ),
            ],
            "validation": [],
        }))

        self.assertIsNotNone(problem)
        self.assertIn("listening_ports evidence", problem)

    def test_generic_assumption_with_matching_tool_event_is_accepted(self):
        import json

        # Deliberately not a web/port case — proves the assumptions
        # mechanism is generic, not special-cased to local web servers.
        agent = core.Agent.__new__(core.Agent)
        agent.workdir = "/home/user/finance-cli"
        agent._read_paths_this_turn = set()
        agent.plan_mode = True
        agent._tool_events = [
            {
                "tool": "run_shell_command",
                "status": "success",
                "args": {"command": "g++ --version"},
                "result": "g++ (Ubuntu 13.2.0) 13.2.0",
            },
        ]

        problem = agent._plan_file_evidence_problem(json.dumps({
            "objective": "Add a new report command to the finance CLI.",
            "files": [],
            "steps": ["Implement the new report command."],
            "validation": [],
            "assumptions": [
                {
                    "claim": "g++ 13 is already installed on this machine",
                    "verified_by": "run_shell_command:g++ --version",
                },
            ],
        }))

        self.assertIsNone(problem)

    def test_generic_assumption_without_matching_tool_event_is_rejected(self):
        import json

        agent = core.Agent.__new__(core.Agent)
        agent.workdir = "/home/user/finance-cli"
        agent._read_paths_this_turn = set()
        agent.plan_mode = True
        agent._tool_events = []

        problem = agent._plan_file_evidence_problem(json.dumps({
            "objective": "Add a new report command to the finance CLI.",
            "files": [],
            "steps": ["Implement the new report command."],
            "validation": [],
            "assumptions": [
                {
                    "claim": "g++ 13 is already installed on this machine",
                    "verified_by": "run_shell_command:g++ --version",
                },
            ],
        }))

        self.assertIsNotNone(problem)
        self.assertIn("g++ 13 is already installed", problem)


if __name__ == "__main__":
    unittest.main()
