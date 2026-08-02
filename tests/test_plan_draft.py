import json
import os
import unittest
from unittest import mock

from agent import core


def valid_plan():
    return {
        "title": "Add feature",
        "objective": "Implement and verify the feature.",
        "files": ["agent/core.py", "tests/test_feature.py"],
        "steps": [
            "Inspect the existing implementation.",
            "Apply the minimum code change.",
        ],
        "validation": [
            {
                "command": "python3 -m unittest discover -s tests",
                "expected": "Exit code 0.",
            }
        ],
        "non_goals": ["Do not refactor unrelated code."],
        "risks": ["Existing callers may rely on current behavior."],
    }


def fenced(payload):
    fence = chr(96) * 3
    return (
        "Plan ready.\n\n"
        + fence
        + "liam-plan\n"
        + json.dumps(payload)
        + "\n"
        + fence
    )


class PlanDraftParserTests(unittest.TestCase):
    def test_complete_plan_is_canonicalized(self):
        canonical, error = core._extract_plan_draft(fenced(valid_plan()))
        self.assertIsNone(error)
        self.assertEqual(json.loads(canonical), valid_plan())

    def test_missing_validation_is_not_ready(self):
        payload = valid_plan()
        del payload["validation"]
        canonical, error = core._extract_plan_draft(fenced(payload))
        self.assertIsNone(canonical)
        self.assertIn("missing required fields: validation", error)

    def test_empty_validation_is_not_ready(self):
        payload = valid_plan()
        payload["validation"] = []
        canonical, error = core._extract_plan_draft(fenced(payload))
        self.assertIsNone(canonical)
        self.assertIn(
            "validation must contain at least one check",
            error,
        )

    def test_unresolved_port_placeholder_is_rejected(self):
        payload = valid_plan()
        payload["validation"][0]["command"] = (
            "curl -fsS http://192.168.0.178:<port>/"
        )

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "contains unresolved placeholder '<port>'",
            error,
        )

    def test_file_changing_step_requires_concrete_files(self):
        payload = valid_plan()
        payload["files"] = []
        payload["steps"] = [
            "Create /var/www/LiamApp01/index.html for the local webpage."
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "files must list the concrete paths",
            error,
        )

    def test_non_goal_cannot_prohibit_plan_execution(self):
        payload = valid_plan()
        payload["non_goals"] = [
            "Creating files, starting servers, or executing the plan."
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "must not prohibit executing or implementing",
            error,
        )

    def test_fileless_non_file_plan_remains_valid(self):
        payload = valid_plan()
        payload["files"] = []
        payload["steps"] = [
            "Restart the already-configured application service."
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertEqual(
            json.loads(canonical)["files"],
            [],
        )

    def test_file_creation_non_goal_conflicts_with_steps(self):
        payload = valid_plan()
        payload["files"] = ["/var/www/LiamApp01/index.html"]
        payload["steps"] = [
            "Create /var/www/LiamApp01/index.html."
        ]
        payload["non_goals"] = [
            "Creating files."
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(canonical)
        self.assertIn(
            "conflict with file-changing implementation steps",
            error,
        )

    def test_scoped_file_non_goal_does_not_conflict(self):
        payload = valid_plan()
        payload["files"] = [
            "/var/www/LiamApp01/index.html",
        ]
        payload["steps"] = [
            "Create /var/www/LiamApp01/index.html.",
        ]
        payload["non_goals"] = [
            "Creating files outside of /var/www/LiamApp01.",
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_server_non_goal_conflicts_with_local_web_plan(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            "Start a local web server on port 8000."
        ]
        payload["non_goals"] = [
            "Starting servers."
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(canonical)
        self.assertIn(
            "conflict with the required local web server",
            error,
        )

    def test_local_web_plan_requires_serving_mechanism(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            "Create the webpage layout and styling."
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(canonical)
        self.assertIn(
            "must include a concrete server command",
            error,
        )

    def test_generic_server_instruction_is_not_concrete(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            "Start a local web server on port 8000 in the background."
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(canonical)
        self.assertIn(
            "must include a concrete server command",
            error,
        )

    def test_foreground_server_instruction_is_rejected(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            "Run `python3 -m http.server 8000 --bind 0.0.0.0`."
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(canonical)
        self.assertIn(
            "remains running during validation",
            error,
        )

    def test_nohup_server_without_redirection_is_rejected(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            (
                "Run `nohup python3 -m http.server 8000 "
                "--bind 0.0.0.0 &`."
            ),
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(canonical)
        self.assertIn(
            "must redirect stdout and stderr",
            error,
        )

    def test_nohup_server_with_redirection_is_valid(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            (
                "Run `nohup python3 -m http.server 8000 "
                "--bind 0.0.0.0 >/dev/null 2>&1 &`."
            ),
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_local_web_plan_with_server_step_is_valid(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            "Create the webpage files.",
            (
                "Run `python3 -m http.server 8000 --bind 0.0.0.0` "
                "as a background process from the webpage directory."
            ),
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_plain_plan_prose_is_not_stored(self):
        canonical, error = core._extract_plan_draft(
            "I still need to inspect another file."
        )
        self.assertIsNone(canonical)
        self.assertIsNone(error)

    @mock.patch.object(core.memory, "create_plan", return_value=41)
    @mock.patch.object(core.memory, "get_latest_plan", return_value=None)
    def test_complete_plan_is_saved_as_draft(
        self,
        get_latest_plan,
        create_plan,
    ):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = True
        agent.session_id = 17
        agent.messages = [{"role": "assistant", "content": "original"}]

        reply = agent._capture_plan_draft(fenced(valid_plan()))

        get_latest_plan.assert_called_once_with(17)
        create_plan.assert_called_once()
        self.assertEqual(create_plan.call_args.args[0], 17)
        self.assertEqual(
            json.loads(create_plan.call_args.args[1]),
            valid_plan(),
        )
        self.assertIn(
            "[Plan draft #41 is ready for approval.]",
            reply,
        )
        self.assertEqual(agent.messages[-1]["content"], reply)

    @mock.patch.object(core.memory, "create_plan")
    @mock.patch.object(core.memory, "get_latest_plan")
    def test_identical_existing_draft_is_reused(
        self,
        get_latest_plan,
        create_plan,
    ):
        canonical, _error = core._extract_plan_draft(
            fenced(valid_plan())
        )
        get_latest_plan.return_value = {
            "id": 52,
            "status": "draft",
            "content": canonical,
        }

        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = True
        agent.session_id = 17
        agent.messages = [{"role": "assistant", "content": "original"}]

        reply = agent._capture_plan_draft(fenced(valid_plan()))

        create_plan.assert_not_called()
        self.assertIn(
            "[Plan draft #52 is ready for approval.]",
            reply,
        )

    @mock.patch.object(core.memory, "create_plan")
    @mock.patch.object(core.memory, "get_latest_plan")
    def test_invalid_block_is_visible_and_not_saved(
        self,
        get_latest_plan,
        create_plan,
    ):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = True
        agent.session_id = 17
        agent.messages = [{"role": "assistant", "content": "original"}]

        fence = chr(96) * 3
        reply = agent._capture_plan_draft(
            fence + "liam-plan\nnot json\n" + fence
        )

        get_latest_plan.assert_not_called()
        create_plan.assert_not_called()
        self.assertIn("[Plan draft not saved:", reply)



    def test_validation_command_cannot_mask_failure_with_echo(self):
        payload = valid_plan()
        payload["validation"] = [
            {
                "command": (
                    "grep -qx 'PLAN TEST' plan-ui-test.txt "
                    "&& echo passed || echo failed"
                ),
                "expected": "Exit code 0 only when the content matches.",
            }
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "validation command masks failure",
            error,
        )

    def test_bare_http_probe_cannot_claim_exact_response(self):
        payload = valid_plan()
        payload["validation"] = [
            {
                "command": (
                    "curl -I http://192.168.0.178:8000"
                ),
                "expected": "HTTP/1.1 200 OK",
            }
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "must assert the exact expected protocol and status",
            error,
        )

    def test_http_probe_with_exact_assertion_is_valid(self):
        payload = valid_plan()
        payload["validation"] = [
            {
                "command": (
                    "curl -sSI http://192.168.0.178:8000 "
                    "| grep -Fq 'HTTP/1.1 200 OK'"
                ),
                "expected": "HTTP/1.1 200 OK",
            }
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_invalid_plan_response_is_corrected_by_one_model_retry(self):
        invalid = valid_plan()
        invalid.pop("non_goals")
        invalid.pop("risks")
        invalid["validation"] = (
            "grep -qx 'PLAN TEST' plan-ui-test.txt "
            "&& echo passed || echo failed"
        )

        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": fenced(invalid),
            },
            {
                "role": "assistant",
                "content": json.dumps(valid_plan()),
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
                return_value=41,
            ) as create_plan,
            mock.patch.dict(
                os.environ,
                {"LIAM_HELPER_OLLAMA_URL": ""},
                clear=False,
            ),
        ):
            agent = core.Agent(
                channel="gui",
                plan_mode=True,
                workdir="/var/www/LiamApp01",
                session_id=17,
            )
            agent.on_status = mock.Mock()
            reply = agent.step(
                "Create a complete plan, but do not execute it."
            )

        self.assertEqual(client.chat.call_count, 2)
        self.assertEqual(
            client.chat.call_args_list[1].kwargs["response_format"],
            core.PLAN_DRAFT_JSON_SCHEMA,
        )
        agent.on_status.assert_called_once()
        self.assertIn(
            "retrying plan formatting (1/2)",
            agent.on_status.call_args.args[0],
        )
        self.assertNotIn(
            "tool selection",
            agent.on_status.call_args.args[0],
        )
        create_plan.assert_called_once()
        self.assertIn(
            "[Plan draft #41 is ready for approval.]",
            reply,
        )
        self.assertNotIn(
            "[Plan draft not saved:",
            reply,
        )

    @mock.patch.object(core.memory, "create_plan", return_value=41)
    @mock.patch.object(core.memory, "get_latest_plan", return_value=None)
    def test_model_plan_notice_is_removed_before_host_notice(
        self,
        get_latest_plan,
        create_plan,
    ):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = True
        agent.session_id = 17
        agent.messages = [{"role": "assistant", "content": "original"}]

        reply = agent._capture_plan_draft(
            fenced(valid_plan())
            + "\n\n[Plan draft #999 is ready for approval.]"
        )

        get_latest_plan.assert_called_once_with(17)
        create_plan.assert_called_once()
        self.assertNotIn(
            "[Plan draft #999 is ready for approval.]",
            reply,
        )
        self.assertEqual(
            reply.count(
                "[Plan draft #41 is ready for approval.]"
            ),
            1,
        )
        self.assertEqual(
            agent.messages[-1]["content"],
            reply,
        )


    def test_plan_format_retry_has_independent_recovery_budget(self):
        invalid = valid_plan()
        invalid.pop("non_goals")
        invalid.pop("risks")

        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": "malformed",
            },
            {
                "role": "assistant",
                "content": fenced(invalid),
            },
            {
                "role": "assistant",
                "content": fenced(valid_plan()),
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
                return_value=52,
            ) as create_plan,
            mock.patch.object(
                core.Agent,
                "_select_recovery_tool_schemas",
                return_value=[],
            ),
            mock.patch.dict(
                os.environ,
                {"LIAM_HELPER_OLLAMA_URL": ""},
                clear=False,
            ),
        ):
            agent = core.Agent(
                channel="gui",
                plan_mode=True,
                workdir="/var/www/LiamApp01",
                session_id=17,
            )
            agent.on_status = mock.Mock()

            reply = agent.step(
                "Create a complete plan, but do not execute it."
            )

        self.assertEqual(client.chat.call_count, 3)
        create_plan.assert_called_once()
        self.assertIn(
            "[Plan draft #52 is ready for approval.]",
            reply,
        )
        self.assertNotIn(
            "[Plan draft not saved:",
            reply,
        )

        statuses = [
            call.args[0]
            for call in agent.on_status.call_args_list
            if call.args
        ]
        self.assertTrue(
            any(
                "retrying plan formatting (1/2)" in status
                for status in statuses
            )
        )


    def test_prose_only_plan_is_corrected_automatically(self):
        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": (
                    "Here is the plan:\n"
                    "1. Inspect the project.\n"
                    "2. Create the webpage.\n"
                    "3. Validate it."
                ),
            },
            {
                "role": "assistant",
                "content": fenced(valid_plan()),
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
                return_value=63,
            ) as create_plan,
            mock.patch.dict(
                os.environ,
                {"LIAM_HELPER_OLLAMA_URL": ""},
                clear=False,
            ),
        ):
            agent = core.Agent(
                channel="gui",
                plan_mode=True,
                workdir="/var/www/LiamApp01",
                session_id=17,
            )
            agent.on_status = mock.Mock()

            reply = agent.step(
                "Make a plan for a local Fluxa webpage."
            )

        self.assertEqual(client.chat.call_count, 2)
        create_plan.assert_called_once()
        self.assertIn(
            "[Plan draft #63 is ready for approval.]",
            reply,
        )
        self.assertNotIn(
            "[Plan draft not saved:",
            reply,
        )

        statuses = [
            call.args[0]
            for call in agent.on_status.call_args_list
            if call.args
        ]
        self.assertTrue(
            any(
                "missing required liam-plan block" in status
                and "retrying plan formatting (1/2)" in status
                for status in statuses
            )
        )


    def test_plan_format_retry_receives_previous_invalid_answer(self):
        failed_answer = (
            "BROKEN PLAN SENTINEL\n"
            "1. Inspect the folder.\n"
            "2. Create the webpage.\n"
            "3. Validate it."
        )

        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": failed_answer,
            },
            {
                "role": "assistant",
                "content": fenced(valid_plan()),
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
                return_value=64,
            ),
            mock.patch.dict(
                os.environ,
                {"LIAM_HELPER_OLLAMA_URL": ""},
                clear=False,
            ),
        ):
            agent = core.Agent(
                channel="gui",
                plan_mode=True,
                workdir="/var/www/LiamApp01",
                session_id=18,
            )
            agent.on_status = mock.Mock()

            reply = agent.step(
                "Make a plan for a local Fluxa webpage."
            )

        self.assertEqual(client.chat.call_count, 2)

        retry_messages = client.chat.call_args_list[1].args[0]
        retry_text = retry_messages[-1]["content"]

        self.assertIn(
            "BROKEN PLAN SENTINEL",
            retry_text,
        )
        self.assertIn(
            "--- BEGIN PREVIOUS ANSWER ---",
            retry_text,
        )
        self.assertIn(
            "--- END PREVIOUS ANSWER ---",
            retry_text,
        )
        self.assertIn(
            "[Plan draft #64 is ready for approval.]",
            reply,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
