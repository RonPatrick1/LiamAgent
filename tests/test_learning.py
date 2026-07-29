import json
import unittest
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

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
    agent.notes_session_id = None
    agent.allowed_tools = None
    agent._tool_events = []
    agent._lesson_uses = []
    agent._current_user_input = ""
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


class MemoryTruthTests(unittest.TestCase):
    def test_explicit_remember_text_is_extracted_without_requiring_copy_paste(self):
        self.assertEqual(
            core._parse_remember_content(
                "Liam, remember to ask Codex about shared playlists."
            ),
            "ask Codex about shared playlists.",
        )
        self.assertEqual(
            core._parse_remember_content(
                "Add the FredPlayer search feature to my notes."
            ),
            "the FredPlayer search feature",
        )

    def test_forget_description_resolves_one_match_but_not_ambiguous_matches(self):
        records = [
            {"id": 7, "content": "Update FredPlayer for Apple Watches"},
            {"id": 8, "content": "Ask Codex about shared playlists"},
        ]
        matched, choices = core._rank_note_matches("FredPlayer update", records)
        self.assertEqual(matched["id"], 7)
        self.assertEqual(choices, [])

        records.append({"id": 9, "content": "Add FredPlayer playlist search"})
        matched, choices = core._rank_note_matches("FredPlayer", records)
        self.assertIsNone(matched)
        self.assertEqual({record["id"] for record in choices}, {7, 9})

    def test_keyword_forget_cannot_bulk_delete_multiple_matches(self):
        connection = mock.MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            (7, "Update FredPlayer for Apple Watches"),
            (9, "Add FredPlayer playlist search"),
        ]
        with mock.patch.object(memory, "_connect", return_value=connection), \
             mock.patch.object(memory, "_ensure_schema"):
            result = memory.forget(keyword="FredPlayer")

        self.assertIn("Multiple notes matched", result)
        self.assertIn("nothing was deleted", result)
        sql_statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertFalse(any(sql.lstrip().upper().startswith("DELETE") for sql in sql_statements))

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    def test_explicit_remember_bypasses_chat_model(self, _match, _save):
        agent = bare_agent(payload=RuntimeError("chat model must not be consulted"))
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_schemas = [
            schema for schema in core.TOOL_SCHEMAS
            if schema["function"]["name"] == "remember"
        ]
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock(return_value="Remembered as #42.")

        reply = agent.step("Remember to ask Codex about shared playlists.")

        self.assertEqual(reply, "Remembered as #42.")
        agent._execute_tool.assert_called_once_with(
            "remember", {"content": "ask Codex about shared playlists."}
        )
        self.assertEqual(agent.client.calls, 0)

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    @mock.patch.object(core.memory, "list_note_records")
    def test_forget_description_deletes_only_resolved_id(self, list_notes, _match, _save):
        list_notes.return_value = [
            {"id": 7, "content": "Update FredPlayer for Apple Watches"},
            {"id": 8, "content": "Ask Codex about shared playlists"},
        ]
        agent = bare_agent(payload=RuntimeError("chat model must not be consulted"))
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_schemas = [
            schema for schema in core.TOOL_SCHEMAS
            if schema["function"]["name"] == "forget"
        ]
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock(return_value="Deleted 1 note(s).")

        reply = agent.step("Forget the FredPlayer update note.")

        self.assertEqual(reply, "Deleted 1 note(s).")
        agent._execute_tool.assert_called_once_with("forget", {"note_id": 7})
        self.assertEqual(agent.client.calls, 0)

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    @mock.patch.object(core.memory, "list_note_records")
    def test_ambiguous_forget_lists_ids_and_deletes_nothing(self, list_notes, _match, _save):
        list_notes.return_value = [
            {"id": 7, "content": "Update FredPlayer for Apple Watches"},
            {"id": 9, "content": "Add FredPlayer playlist search"},
        ]
        agent = bare_agent(payload=RuntimeError("chat model must not be consulted"))
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_schemas = [
            schema for schema in core.TOOL_SCHEMAS
            if schema["function"]["name"] == "forget"
        ]
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock()

        reply = agent.step("Forget the FredPlayer note.")

        self.assertIn("did not delete any", reply)
        self.assertIn("#7", reply)
        self.assertIn("#9", reply)
        agent._execute_tool.assert_not_called()
        self.assertEqual(agent.client.calls, 0)

    def test_only_current_explicit_commands_authorize_note_mutation(self):
        self.assertTrue(core.Agent._explicit_note_action_requested(
            "forget", "Liam: please forget the note that says update FredPlayer."
        ))
        self.assertTrue(core.Agent._explicit_note_action_requested(
            "remember", "Liam, remember to ask Codex about shared playlists."
        ))
        self.assertTrue(core.Agent._explicit_note_action_requested(
            "remember", "Don't forget to ask Codex about shared playlists."
        ))
        self.assertFalse(core.Agent._explicit_note_action_requested(
            "forget", "Didn't I ask you to forget the FredPlayer note?"
        ))
        self.assertFalse(core.Agent._explicit_note_action_requested(
            "forget",
            "I didn't ask you to forget anything. Earlier I said: Liam: forget the old note.",
        ))
        self.assertFalse(core.Agent._explicit_note_action_requested(
            "remember", "Do you remember when we discussed shared playlists?"
        ))

    def test_forget_tool_is_blocked_for_a_complaint_that_quotes_old_command(self):
        agent = bare_agent()
        agent._read_paths_this_turn = set()
        agent._current_user_input = (
            "I didn't ask you to forget anything. Earlier I said: "
            "Liam: forget the note that says update FredPlayer."
        )

        result = agent._run_tool("forget", {"keyword": "FredPlayer"})

        self.assertIn("does not explicitly request", result)

    def test_false_memory_claim_is_corrected_and_recorded(self):
        agent = bare_agent()
        agent._record_intervention = mock.Mock()

        result = agent._note_unperformed_memory_actions(
            "I have now forgotten the following notes as requested.", []
        )

        self.assertIn("no forget tool successfully deleted", result)
        agent._record_intervention.assert_called_once_with(
            "memory_claim_without_tool", "forget",
            "The model claimed saved notes were deleted, but no forget call succeeded this turn.",
        )

    def test_echoed_memory_warning_is_replaced_instead_of_duplicated(self):
        agent = bare_agent()
        agent._record_intervention = mock.Mock()
        stale = (
            "I have now forgotten the note.\n\n"
            "[Note: no forget tool successfully deleted a saved note this turn; "
            "any claimed deletion is not authoritative.]"
        )

        result = agent._note_unperformed_memory_actions(stale, [])

        self.assertEqual(result.count("[Note:"), 1)
        self.assertEqual(result.lower().count("no forget tool successfully"), 1)

    def test_successful_memory_tool_makes_matching_claim_authoritative(self):
        agent = bare_agent()
        agent._record_intervention = mock.Mock()
        content = "I have now forgotten that note."

        result = agent._note_unperformed_memory_actions(
            content, [("forget", "Deleted 1 note(s).")]
        )

        self.assertEqual(result, content)
        agent._record_intervention.assert_not_called()

    def test_model_cannot_fabricate_host_lesson_notice(self):
        agent = bare_agent()
        agent.messages = []
        agent._record_auto_lessons = mock.Mock()
        agent._evaluate_lesson_uses = mock.Mock()

        result = agent._finalize_learning(
            "Answer.\n\n[I queued that feedback as lesson candidate #17 for owner review; "
            "it is not active yet.]",
            None,
        )

        self.assertEqual(result, "Answer.")

    def test_fake_lesson_notice_is_removed_even_after_a_false_memory_claim(self):
        agent = bare_agent()
        agent.messages = []
        agent._record_auto_lessons = mock.Mock()
        agent._evaluate_lesson_uses = mock.Mock()
        agent._record_intervention = mock.Mock()
        content = agent._note_unperformed_memory_actions(
            "I have now forgotten the note.\n\n"
            "[I queued that feedback as lesson candidate #17 for owner review.]",
            [],
        )

        result = agent._finalize_learning(content, None)

        self.assertNotIn("lesson candidate #17", result)
        self.assertIn("no forget tool successfully deleted", result)

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    def test_direct_note_recall_bypasses_chat_model(self, _match, _save_message):
        agent = bare_agent(payload=RuntimeError("chat model must not be consulted"))
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_schemas = [
            schema for schema in core.TOOL_SCHEMAS
            if schema["function"]["name"] == "recall_notes"
        ]
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock(return_value="#15: Update FredPlayer")

        reply = agent.step("What notes do you remember?")

        self.assertEqual(reply, "#15: Update FredPlayer")
        agent._execute_tool.assert_called_once_with("recall_notes", {})
        self.assertEqual(agent.client.calls, 0)


class RoutineRoutingTests(unittest.TestCase):
    NOW = datetime(2026, 7, 24, 9, 24, tzinfo=ZoneInfo("America/Detroit"))

    def test_original_failed_matrix_request_parses_as_one_time_schedule(self):
        parsed = core.Agent._parse_schedule_request(
            "can you schedule a test routine to run at 9:25a today (in a few minutes) "
            "to say how handsome Ron is in this chat?",
            now=self.NOW,
        )

        self.assertEqual(parsed["schedule_kind"], "once")
        self.assertEqual(parsed["schedule_value"], "2026-07-24 09:25:00")
        self.assertIn("Original request:", parsed["prompt"])

    def test_daily_and_hourly_schedules_parse_deterministically(self):
        daily = core.Agent._parse_schedule_request(
            "Remind me every day at 8:05pm to check the porch.", now=self.NOW,
        )
        hourly = core.Agent._parse_schedule_request(
            "Schedule a routine every 4 hours to check the server.", now=self.NOW,
        )
        minutely = core.Agent._parse_schedule_request(
            "Send me a test message every 5 minutes.", now=self.NOW,
        )
        abbreviated = core.Agent._parse_schedule_request(
            "Send me a test message every 15 mins.", now=self.NOW,
        )
        conversational = core.Agent._parse_schedule_request(
            "liam: tell me I'm handsome every 5 minutes", now=self.NOW,
        )

        self.assertEqual(
            (daily["schedule_kind"], daily["schedule_value"]), ("daily", "20:05")
        )
        self.assertEqual(
            (hourly["schedule_kind"], hourly["schedule_value"]), ("hourly", "4")
        )
        self.assertEqual(
            (minutely["schedule_kind"], minutely["schedule_value"]),
            ("minutely", "5"),
        )
        self.assertEqual(
            (abbreviated["schedule_kind"], abbreviated["schedule_value"]),
            ("minutely", "15"),
        )
        self.assertEqual(
            (conversational["schedule_kind"], conversational["schedule_value"]),
            ("minutely", "5"),
        )

    def test_schedule_complaint_or_quote_is_not_a_new_schedule(self):
        self.assertIsNone(core.Agent._parse_schedule_request(
            "Why didn't this work? Earlier I said: schedule it at 9:25 today.",
            now=self.NOW,
        ))
        self.assertIsNone(core.Agent._parse_schedule_request(
            "Can you tell me what the weather will be at 8pm?", now=self.NOW,
        ))
        self.assertIsNone(core.Agent._parse_schedule_request(
            "Tell me why that job runs every 5 minutes.", now=self.NOW,
        ))

    def test_natural_cancel_request_extracts_description_or_id(self):
        self.assertEqual(
            core._parse_cancel_routine_target(
                "liam: stop telling me I'm handsome every 5 minutes"
            ),
            {"query": "telling me I'm handsome every 5 minutes"},
        )
        self.assertEqual(
            core._parse_cancel_routine_target("cancel routine #5"),
            {"routine_id": 5},
        )
        self.assertIsNone(core._parse_cancel_routine_target(
            "delete the file named routine.txt"
        ))

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    def test_explicit_schedule_bypasses_chat_model(self, _match, _save_message):
        agent = bare_agent(payload=RuntimeError("chat model must not be consulted"))
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_schemas = [
            schema for schema in core.TOOL_SCHEMAS
            if schema["function"]["name"] == "schedule_routine"
        ]
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock(return_value=(
            "Scheduled routine #8: test — runs once at 2026-07-29 11:00:00, in this thread."
        ))

        reply = agent.step("Schedule a test routine at 11:00am today to say hello.")

        self.assertTrue(reply.startswith("Scheduled routine #8:"))
        self.assertEqual(agent._execute_tool.call_args.args[0], "schedule_routine")
        self.assertEqual(agent.client.calls, 0)

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    def test_tell_me_recurring_schedule_bypasses_chat_model(self, _match, _save_message):
        agent = bare_agent(payload=RuntimeError("chat model must not be consulted"))
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_schemas = [
            schema for schema in core.TOOL_SCHEMAS
            if schema["function"]["name"] == "schedule_routine"
        ]
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock(return_value=(
            "Scheduled routine #9 — runs every 5 minute(s), in this thread."
        ))

        reply = agent.step("liam: tell me I'm handsome every 5 minutes")

        self.assertTrue(reply.startswith("Scheduled routine #9"))
        agent._execute_tool.assert_called_once()
        name, args = agent._execute_tool.call_args.args
        self.assertEqual(name, "schedule_routine")
        self.assertEqual((args["schedule_kind"], args["schedule_value"]), ("minutely", "5"))
        self.assertEqual(agent.client.calls, 0)

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    @mock.patch.object(core.routines, "list_routines")
    def test_natural_cancellation_resolves_and_calls_exact_routine(
        self, list_routines, _match, _save_message,
    ):
        list_routines.return_value = [{
            "id": 5,
            "session_id": 17,
            "enabled": True,
            "schedule_kind": "minutely",
            "schedule_value": "5",
            "prompt": (
                "This is the scheduled execution time. Original request:\n"
                "tell me I'm handsome every 5 minutes"
            ),
        }]
        agent = bare_agent(payload=RuntimeError("chat model must not be consulted"))
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_schemas = [
            schema for schema in core.TOOL_SCHEMAS
            if schema["function"]["name"] == "cancel_routine"
        ]
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock(return_value="Cancelled routine #5.")

        reply = agent.step("liam: stop telling me I'm handsome every 5 minutes")

        self.assertEqual(reply, "Cancelled routine #5.")
        agent._execute_tool.assert_called_once_with("cancel_routine", {"routine_id": 5})
        self.assertEqual(agent.client.calls, 0)

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    @mock.patch.object(core.routines, "list_routines")
    def test_ambiguous_cancellation_lists_routines_and_cancels_none(
        self, list_routines, _match, _save_message,
    ):
        list_routines.return_value = [
            {
                "id": 5, "session_id": 17, "enabled": True,
                "schedule_kind": "minutely", "schedule_value": "5",
                "prompt": "tell me I'm handsome every 5 minutes",
            },
            {
                "id": 6, "session_id": 17, "enabled": True,
                "schedule_kind": "minutely", "schedule_value": "5",
                "prompt": "tell me the server status every 5 minutes",
            },
        ]
        agent = bare_agent(payload=RuntimeError("chat model must not be consulted"))
        agent.messages = [{"role": "system", "content": "system"}]
        agent.tool_schemas = [
            schema for schema in core.TOOL_SCHEMAS
            if schema["function"]["name"] == "cancel_routine"
        ]
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock()

        reply = agent.step("Cancel the routine every 5 minutes")

        self.assertIn("cancelled none", reply)
        self.assertIn("#5", reply)
        self.assertIn("#6", reply)
        agent._execute_tool.assert_not_called()
        self.assertEqual(agent.client.calls, 0)

    def test_false_scheduled_claim_is_corrected_and_recorded(self):
        agent = bare_agent()
        agent._record_intervention = mock.Mock()

        reply = agent._note_unperformed_schedule(
            "I've scheduled the routine for 9:25 AM.", []
        )

        self.assertEqual(
            reply,
            "I couldn't create that routine because no timer was actually created. "
            "Nothing has been scheduled.",
        )
        self.assertNotIn("I've scheduled", reply)
        agent._record_intervention.assert_called_once()

    def test_echoed_schedule_warning_is_replaced_instead_of_duplicated(self):
        agent = bare_agent()
        agent._record_intervention = mock.Mock()
        stale = (
            "I've scheduled it.\n\n"
            "[Note: no schedule_routine call successfully created a timer this turn; "
            "any scheduling claim above is false.]"
        )

        result = agent._note_unperformed_schedule(stale, [])

        self.assertNotIn("[Note:", result)
        self.assertNotIn("I've scheduled", result)
        self.assertIn("Nothing has been scheduled", result)

    def test_false_cancellation_claim_is_replaced_and_recorded(self):
        agent = bare_agent()
        agent._record_intervention = mock.Mock()

        reply = agent._note_unperformed_cancellation(
            "I have canceled the routine that told you you're handsome.", []
        )

        self.assertNotIn("I have canceled", reply)
        self.assertIn("no scheduled timer was removed", reply)
        self.assertIn("still be active", reply)
        agent._record_intervention.assert_called_once_with(
            "cancel_claim_without_tool", "cancel_routine",
            "The model claimed a routine was cancelled, but no cancel_routine call succeeded this turn.",
        )


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
