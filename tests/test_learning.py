import json
import unittest
from unittest import mock

from agent import core, memory


class FakeClient:
    def __init__(self, payload=None):
        self.payload = payload
        self.calls = 0

    def chat(self, _messages, tools=None):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return {"content": self.payload or "not json"}


def bare_agent(*, owner=True, learning=True, payload=None):
    agent = core.Agent.__new__(core.Agent)
    agent.client = FakeClient(payload)
    agent.learning_enabled = learning
    agent.is_owner = owner
    agent.channel = "gui" if owner else "matrix"
    agent.actor_id = "local-owner" if owner else "@guest:example"
    agent.workdir = "/tmp/liam-project"
    agent.session_id = 17
    agent.allowed_tools = None
    agent._tool_events = []
    agent._lesson_uses = []
    return agent


def tool_event(tool, status, reason, result, *, args=None, family=None,
               transient=False):
    return {
        "tool": tool,
        "args": dict(args or {}),
        "result": result,
        "status": status,
        "reason": reason,
        "transient": transient,
        "validation": family is not None,
        "family": family,
        "signature": f"{reason}:evidence" if status in {"failure", "noop"} else None,
    }


class ToolOutcomeTests(unittest.TestCase):
    def test_nonzero_test_command_is_a_validation_failure(self):
        agent = bare_agent()
        outcome = agent._classify_tool_outcome(
            "run_shell_command", {"command": "pytest -q"},
            "1 failed\n[exit code: 1]",
        )
        self.assertEqual(outcome["status"], "failure")
        self.assertEqual(outcome["reason"], "nonzero_exit")
        self.assertTrue(outcome["validation"])
        self.assertEqual(outcome["family"], "pytest")

    def test_identical_write_is_a_noop(self):
        agent = bare_agent()
        outcome = agent._classify_tool_outcome(
            "write_file", {"path": "a.py", "content": "same"},
            "File is byte-for-byte identical; nothing would actually change.",
        )
        self.assertEqual((outcome["status"], outcome["reason"]), ("noop", "no_change"))

    def test_external_outage_is_not_learnable(self):
        agent = bare_agent()
        outcome = agent._classify_tool_outcome(
            "web_search", {"query": "x"}, "Web search failed: connection refused",
        )
        self.assertEqual(outcome["status"], "transient")
        self.assertTrue(outcome["transient"])

    def test_git_exit_code_is_classified_deterministically(self):
        agent = bare_agent()
        outcome = agent._classify_tool_outcome(
            "git_add", {"path": "missing.txt"},
            "fatal: pathspec did not match any files\n[exit code: 128]",
        )
        self.assertEqual((outcome["status"], outcome["reason"]),
                         ("failure", "nonzero_exit"))

    def test_empty_search_result_has_a_specific_reason(self):
        agent = bare_agent()
        outcome = agent._classify_tool_outcome(
            "image_search", {"query": "impossible query"}, "No images found.",
        )
        self.assertEqual((outcome["status"], outcome["reason"]),
                         ("failure", "empty_result"))

    def test_invalid_fetch_url_is_not_reported_as_success(self):
        agent = bare_agent()
        outcome = agent._classify_tool_outcome(
            "fetch_url", {"url": "notes.txt"},
            "fetch_url only works on real http(s) webpages, not 'notes.txt'.",
        )
        self.assertEqual((outcome["status"], outcome["reason"]),
                         ("failure", "invalid_arguments"))


class ImageRoutingTests(unittest.TestCase):
    def test_explicit_creation_requests_are_distinguished_from_questions_and_search(self):
        self.assertTrue(core.Agent._is_direct_image_request(
            "create an image with bad eye balls"
        ))
        self.assertTrue(core.Agent._is_direct_image_request(
            "Could you make me a picture of a magic goat?"
        ))
        self.assertFalse(core.Agent._is_direct_image_request(
            "How do I generate an image?"
        ))
        self.assertFalse(core.Agent._is_direct_image_request(
            "Show me a picture of a mountain"
        ))
        self.assertFalse(core.Agent._is_direct_image_request(
            "Create a Python program that generates images"
        ))

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    def test_direct_image_request_bypasses_chat_model(self, _match, save_message):
        agent = bare_agent(payload=RuntimeError("chat model must not be consulted"))
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_schemas = [
            schema for schema in core.TOOL_SCHEMAS
            if schema["function"]["name"] == "generate_image"
        ]
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock(return_value=(
            "Generated image saved. ![bad eyeballs](/tmp/real-image.png)"
        ))

        reply = agent.step("create an image with bad eye balls")

        self.assertIn("/tmp/real-image.png", reply)
        self.assertNotIn("unable", reply.lower())
        agent._execute_tool.assert_called_once_with(
            "generate_image", {"prompt": "create an image with bad eye balls"}
        )
        self.assertEqual(agent.client.calls, 0)
        self.assertEqual(save_message.call_count, 2)

    def test_refusal_fallback_replaces_refusal_and_records_contract_failure(self):
        agent = bare_agent()
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock(return_value=(
            "Generated image saved. ![bad eyeballs](/tmp/fallback-image.png)"
        ))

        content, results = agent._auto_generate_missing_image(
            "create an image with bad eye balls",
            "I'm unable to fulfill that request because it is inappropriate content.",
            [],
        )

        self.assertIn("/tmp/fallback-image.png", content)
        self.assertNotIn("unable", content.lower())
        self.assertEqual(results[0][0], "generate_image")
        self.assertEqual(agent._tool_events[0]["reason"], "available_capability_refusal")


class AutomaticLessonTests(unittest.TestCase):
    @mock.patch.object(core.memory, "upsert_lesson")
    def test_verified_failure_then_success_activates_workspace_lesson(self, upsert):
        upsert.return_value = {"id": 1, "status": "active"}
        agent = bare_agent()
        agent._tool_events = [
            tool_event(
                "run_shell_command", "failure", "nonzero_exit",
                "error: missing import\n[exit code: 1]",
                args={"command": "pytest -q"}, family="pytest",
            ),
            tool_event(
                "run_shell_command", "success", "completed",
                "8 passed\n[exit code: 0]",
                args={"command": "pytest -q"}, family="pytest",
            ),
        ]

        agent._record_auto_lessons("The tests now pass.")

        upsert.assert_called_once()
        kwargs = upsert.call_args.kwargs
        self.assertEqual(kwargs["status"], "active")
        self.assertEqual(kwargs["origin"], "verified_recovery")
        self.assertEqual(kwargs["detector"], "validation_recovery")
        self.assertEqual((kwargs["scope_kind"], kwargs["scope_value"]),
                         ("workspace", "/tmp/liam-project"))

    @mock.patch.object(core.memory, "upsert_lesson")
    def test_failure_without_recovery_does_not_teach(self, upsert):
        agent = bare_agent()
        agent._tool_events = [
            tool_event("edit_file", "failure", "edit_text_not_found", "old_string not found"),
        ]
        agent._record_auto_lessons("I could not apply the edit.")
        upsert.assert_not_called()

    @mock.patch.object(core.memory, "upsert_lesson")
    def test_transient_failure_does_not_teach(self, upsert):
        agent = bare_agent()
        agent._tool_events = [
            tool_event(
                "web_search", "transient", "external_dependency",
                "connection refused", transient=True,
            ),
        ]
        agent._record_auto_lessons("The service is unavailable.")
        upsert.assert_not_called()

    @mock.patch.object(core.memory, "upsert_lesson")
    def test_host_contract_violation_is_active_without_model_judgment(self, upsert):
        upsert.return_value = {"id": 2, "status": "active"}
        agent = bare_agent(payload=RuntimeError("synthesis must not run"))
        agent._tool_events = [
            tool_event(
                "generate_image", "failure", "image_claim_without_tool",
                "The model claimed an image without calling the tool.",
            ),
        ]
        agent._record_auto_lessons("Here is the image.")
        upsert.assert_called_once()
        self.assertEqual(upsert.call_args.kwargs["origin"], "contract_violation")
        self.assertEqual(upsert.call_args.kwargs["scope_value"], "generate_image")
        self.assertEqual(agent.client.calls, 0)

    @mock.patch.object(core.memory, "upsert_lesson")
    def test_false_success_claim_is_detected(self, upsert):
        upsert.return_value = {"id": 3, "status": "active"}
        agent = bare_agent()
        agent._tool_events = [
            tool_event(
                "write_file", "failure", "tool_error",
                "Error: permission denied", args={"path": "a.py"},
            ),
        ]
        agent._record_auto_lessons("Done — I successfully saved the file.")
        self.assertEqual(upsert.call_args.kwargs["detector"], "false_success_report")


class ChatFeedbackTests(unittest.TestCase):
    def feedback_payload(self, *, explicit=True, confidence=0.97):
        return json.dumps({
            "actionable": True,
            "explicit": explicit,
            "confidence": confidence,
            "keywords": ["generated image", "image tool"],
            "lesson": "Always call generate_image before claiming an image was generated.",
            "scope_kind": "tool",
            "scope_value": "generate_image",
        })

    @mock.patch.object(core.memory, "upsert_lesson")
    def test_explicit_high_confidence_owner_feedback_is_active(self, upsert):
        upsert.return_value = {"id": 11, "status": "active", "created_new": True}
        agent = bare_agent(payload=self.feedback_payload())
        notice = agent._capture_chat_feedback(
            "Here is a generated image.",
            "That was wrong. Next time always call the image tool first.",
        )
        self.assertEqual(upsert.call_args.kwargs["status"], "active")
        self.assertEqual(upsert.call_args.kwargs["origin"], "owner_feedback")
        self.assertIn("learned", notice)

    @mock.patch.object(core.memory, "upsert_lesson")
    def test_nonowner_feedback_is_pending(self, upsert):
        upsert.return_value = {"id": 12, "status": "pending", "created_new": True}
        agent = bare_agent(owner=False, payload=self.feedback_payload())
        notice = agent._capture_chat_feedback(
            "Here is a generated image.",
            "Wrong. Next time always call the image tool first.",
        )
        self.assertEqual(upsert.call_args.kwargs["status"], "pending")
        self.assertEqual(upsert.call_args.kwargs["origin"], "participant_feedback")
        self.assertIn("owner review", notice)

    @mock.patch.object(core.memory, "upsert_lesson")
    def test_ambiguous_owner_feedback_is_pending(self, upsert):
        upsert.return_value = {"id": 13, "status": "pending", "created_new": True}
        agent = bare_agent(payload=self.feedback_payload(explicit=False, confidence=0.82))
        agent._capture_chat_feedback(
            "I used a web result.", "Actually that source seems wrong.",
        )
        self.assertEqual(upsert.call_args.kwargs["status"], "pending")

    @mock.patch.object(core.memory, "upsert_lesson")
    def test_low_confidence_feedback_is_ignored(self, upsert):
        agent = bare_agent(payload=self.feedback_payload(confidence=0.40))
        notice = agent._capture_chat_feedback("Answer", "Actually, maybe not.")
        self.assertIsNone(notice)
        upsert.assert_not_called()

    @mock.patch.object(core.memory, "quarantine_latest_feedback_lesson", return_value=21)
    def test_owner_can_undo_latest_chat_lesson(self, quarantine):
        agent = bare_agent(payload=RuntimeError("classifier should not run"))
        notice = agent._capture_chat_feedback("Answer", "Don't learn that correction.")
        quarantine.assert_called_once_with(17)
        self.assertIn("quarantined lesson #21", notice)
        self.assertEqual(agent.client.calls, 0)

    @mock.patch.object(core.memory, "upsert_lesson")
    def test_disabled_channel_cannot_learn(self, upsert):
        agent = bare_agent(learning=False, payload=RuntimeError("classifier should not run"))
        notice = agent._capture_chat_feedback("Answer", "Wrong. Always do X instead.")
        self.assertIsNone(notice)
        upsert.assert_not_called()
        self.assertEqual(agent.client.calls, 0)


class LessonMatchingTests(unittest.TestCase):
    def record(self, lesson_id, keywords, scope_kind="global", scope_value=None,
               status="active"):
        return {
            "id": lesson_id,
            "keywords": keywords,
            "lesson": f"Lesson {lesson_id} is long enough.",
            "status": status,
            "scope_kind": scope_kind,
            "scope_value": scope_value,
            "updated_at": lesson_id,
        }

    def test_word_boundaries_prevent_partial_keyword_matches(self):
        records = [self.record(1, "cat")]
        self.assertEqual(memory.select_matching_lesson_records(records, "concatenate"), [])
        self.assertEqual(
            [item["id"] for item in memory.select_matching_lesson_records(records, "the cat")],
            [1],
        )

    def test_scope_status_and_limit_are_enforced(self):
        records = [
            self.record(1, "build"),
            self.record(2, "build,test", "workspace", "/tmp/liam-project"),
            self.record(3, "build", "channel", "matrix"),
            self.record(4, "build", "tool", "run_shell_command"),
            self.record(5, "build", status="pending"),
        ]
        selected = memory.select_matching_lesson_records(
            records, "build and test", workspace="/tmp/liam-project",
            channel="gui", available_tools={"run_shell_command"}, limit=3,
        )
        self.assertEqual([item["id"] for item in selected], [2, 4, 1])

    def test_fingerprint_is_stable_and_case_insensitive(self):
        first = memory.lesson_fingerprint("Detector", " Failure  Signature ")
        second = memory.lesson_fingerprint("Detector", "failure signature")
        self.assertEqual(first, second)


class LessonEffectivenessTests(unittest.TestCase):
    class Cursor:
        def __init__(self):
            self.statements = []
            self.last_sql = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params=None):
            self.last_sql = " ".join(sql.split())
            self.statements.append((self.last_sql, params))

        def fetchone(self):
            if self.last_sql.startswith("SELECT lesson_id FROM lesson_uses"):
                return (44,)
            return None

    class Connection:
        def __init__(self, cursor):
            self._cursor = cursor

        def cursor(self):
            return self._cursor

        def close(self):
            pass

    def test_failure_increments_counter_and_quarantines_only_auto_origins_at_two(self):
        cursor = self.Cursor()
        connection = self.Connection(cursor)
        with mock.patch.object(memory, "_connect", return_value=connection), \
             mock.patch.object(memory, "_ensure_schema"), \
             mock.patch.object(memory, "get_lesson", return_value={"id": 44}):
            record = memory.resolve_lesson_use(9, "failure", "same-error")

        self.assertEqual(record, {"id": 44})
        sql = "\n".join(statement for statement, _params in cursor.statements)
        self.assertIn("consecutive_failure_count = consecutive_failure_count + 1", sql)
        self.assertIn("origin IN ('verified_recovery', 'contract_violation')", sql)
        self.assertIn("consecutive_failure_count >= 2", sql)

    def test_success_resets_consecutive_failures(self):
        cursor = self.Cursor()
        connection = self.Connection(cursor)
        with mock.patch.object(memory, "_connect", return_value=connection), \
             mock.patch.object(memory, "_ensure_schema"), \
             mock.patch.object(memory, "get_lesson", return_value={"id": 44}):
            memory.resolve_lesson_use(9, "success")

        sql = "\n".join(statement for statement, _params in cursor.statements)
        self.assertIn("success_count = success_count + 1", sql)
        self.assertIn("consecutive_failure_count = 0", sql)


if __name__ == "__main__":
    unittest.main()
