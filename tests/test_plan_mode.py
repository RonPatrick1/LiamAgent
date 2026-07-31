import os
import tempfile
import unittest
from unittest import mock

from agent import core


class PlanModeTests(unittest.TestCase):
    def _build_agent(self, plan_mode=False, workdir=None):
        patches = [
            mock.patch.object(core, "OllamaClient", return_value=mock.Mock()),
            mock.patch.object(core.memory, "load_recent_notes", return_value=[]),
            mock.patch.object(core.memory, "load_recent_messages", return_value=[]),
            mock.patch.dict(
                os.environ,
                {"LIAM_HELPER_OLLAMA_URL": ""},
                clear=False,
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        return core.Agent(
            channel="gui",
            workdir=workdir,
            plan_mode=plan_mode,
        )

    @staticmethod
    def _offered_names(agent):
        return {
            schema["function"]["name"]
            for schema in agent.tool_schemas
        }

    def test_normal_mode_retains_existing_gui_tool_set(self):
        agent = self._build_agent()

        self.assertFalse(agent.plan_mode)
        self.assertEqual(
            self._offered_names(agent),
            set(core.TOOL_IMPL) - {"propose_lesson"},
        )

    def test_plan_mode_advertises_only_explicit_allowlist(self):
        agent = self._build_agent(plan_mode=True)

        self.assertTrue(agent.plan_mode)
        self.assertEqual(
            self._offered_names(agent),
            core.PLAN_MODE_ALLOWED_TOOLS,
        )
        self.assertLessEqual(
            core.PLAN_MODE_ALLOWED_TOOLS,
            set(core.TOOL_IMPL),
        )

    def test_plan_allowlist_excludes_generic_and_remote_shells(self):
        self.assertNotIn(
            "run_shell_command",
            core.PLAN_MODE_ALLOWED_TOOLS,
        )
        self.assertNotIn(
            "ssh_run_command",
            core.PLAN_MODE_ALLOWED_TOOLS,
        )
        self.assertNotIn(
            "ssh_list_hosts",
            core.PLAN_MODE_ALLOWED_TOOLS,
        )

    def test_mutating_tool_is_blocked_before_confirmation_or_execution(self):
        agent = self._build_agent(plan_mode=True)
        agent.on_confirm = mock.Mock(return_value=True)
        implementation = mock.Mock(return_value="unexpected write")

        with mock.patch.dict(
            core.TOOL_IMPL,
            {"write_file": implementation},
        ):
            result = agent._run_tool(
                "write_file",
                {"path": "x.txt", "content": "changed"},
            )

        self.assertIn("unavailable in Plan mode", result)
        self.assertIn("no action was performed", result)
        implementation.assert_not_called()
        agent.on_confirm.assert_not_called()

    def test_permitted_read_only_tool_still_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            test_path = os.path.join(directory, "example.txt")
            with open(test_path, "w") as handle:
                handle.write("observed repository content")

            agent = self._build_agent(
                plan_mode=True,
                workdir=directory,
            )
            result = agent._run_tool(
                "read_file",
                {"path": "example.txt"},
            )

        self.assertEqual(result, "observed repository content")

    def test_plan_instructions_are_in_initial_system_message(self):
        agent = self._build_agent(plan_mode=True)

        self.assertEqual(agent.messages[0]["role"], "system")
        self.assertIn(
            "PLAN MODE IS ACTIVE",
            agent.messages[0]["content"],
        )
        self.assertIn(
            "Identify the files and",
            agent.messages[0]["content"],
        )
        self.assertFalse(
            any(
                "PLAN MODE IS ACTIVE" in message.get("content", "")
                for message in agent.messages[1:]
            )
        )

    def test_existing_constructor_usage_defaults_to_normal_mode(self):
        agent = self._build_agent()

        self.assertIs(agent.plan_mode, False)
        self.assertTrue(agent.learning_enabled)


    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(
        core.memory,
        "match_lesson_records",
        return_value=[],
    )
    def test_plan_mode_skips_mutating_deterministic_routes(
        self,
        _match_lessons,
        _save_message,
    ):
        requests = [
            "On alien, run `uname -a`.",
            "Schedule a test routine at 11:00am today to say hello.",
            "Cancel routine #8.",
            "Remember to ask about shared playlists.",
            "Forget note #42.",
            "What notes do you remember?",
            "Create an image of a blue robot.",
        ]

        for request in requests:
            with self.subTest(request=request):
                agent = self._build_agent(plan_mode=True)
                agent.client = mock.Mock()
                agent.client.chat.return_value = {
                    "content": "Plan-only response",
                }
                agent.on_tool_call = mock.Mock()
                agent._execute_tool = mock.Mock(
                    side_effect=AssertionError(
                        "Plan mode must not execute a deterministic action"
                    )
                )

                reply = agent.step(request)

                self.assertEqual(reply, "Plan-only response")
                agent._execute_tool.assert_not_called()
                agent.on_tool_call.assert_not_called()
                agent.client.chat.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
