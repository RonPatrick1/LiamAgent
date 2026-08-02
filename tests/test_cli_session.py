import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import LiamAgent as cli


class CliSessionTests(unittest.TestCase):
    def test_session_id_selects_exact_persisted_thread(self):
        session = {
            "id": 47,
            "title": "Fluxa",
            "folder_path": "/var/www/LiamApp01",
            "plan_mode": True,
        }
        args = SimpleNamespace(
            session_id=47,
            workdir=None,
        )

        with (
            mock.patch.object(
                cli.memory,
                "get_session",
                return_value=session,
            ) as get_session,
            mock.patch.object(
                cli.memory,
                "get_or_create_session",
            ) as get_or_create,
        ):
            result = cli._resolve_cli_session(args)

        self.assertEqual(result, session)
        get_session.assert_called_once_with(47)
        get_or_create.assert_not_called()

    def test_workdir_reuses_folder_session(self):
        with tempfile.TemporaryDirectory() as directory:
            resolved = os.path.abspath(directory)
            session = {
                "id": 19,
                "title": "Workspace",
                "folder_path": resolved,
                "plan_mode": False,
            }
            args = SimpleNamespace(
                session_id=None,
                workdir=directory,
            )

            with (
                mock.patch.object(
                    cli.memory,
                    "get_or_create_session",
                    return_value=19,
                ) as get_or_create,
                mock.patch.object(
                    cli.memory,
                    "get_session",
                    return_value=session,
                ) as get_session,
            ):
                result = cli._resolve_cli_session(args)

            self.assertEqual(result, session)
            get_or_create.assert_called_once_with(resolved)
            get_session.assert_called_once_with(19)

    def test_one_shot_prompt_uses_gui_session_state(self):
        session = {
            "id": 47,
            "title": "Fluxa",
            "folder_path": "/var/www/LiamApp01",
            "plan_mode": True,
        }
        agent = mock.Mock()
        agent.plan_mode = True
        agent._tool_events = []
        agent.step.return_value = "CLI response"

        saved = {
            "model": "test-model",
            "auto_confirm": True,
            "custom_instructions": "shared instructions",
        }

        with (
            mock.patch.object(
                cli.sys,
                "argv",
                [
                    "LiamAgent.py",
                    "--session-id",
                    "47",
                    "--prompt",
                    "Make a plan.",
                ],
            ),
            mock.patch.object(
                cli.liam_settings,
                "load",
                return_value=saved,
            ),
            mock.patch.object(
                cli.memory,
                "get_session",
                return_value=session,
            ),
            mock.patch.object(
                cli.memory,
                "list_session_folders",
                return_value=[
                    {"folder_path": "/var/www/shared"},
                ],
            ),
            mock.patch.object(
                cli,
                "Agent",
                return_value=agent,
            ) as agent_class,
            mock.patch.object(
                cli,
                "ensure_visible_reply",
                return_value="CLI response",
            ),
            mock.patch("builtins.print"),
        ):
            cli.main()

        kwargs = agent_class.call_args.kwargs
        self.assertEqual(kwargs["session_id"], 47)
        self.assertEqual(
            kwargs["workdir"],
            "/var/www/LiamApp01",
        )
        self.assertEqual(
            kwargs["extra_folders"],
            ["/var/www/shared"],
        )
        self.assertEqual(
            kwargs["custom_instructions"],
            "shared instructions",
        )
        self.assertTrue(kwargs["plan_mode"])
        agent.step.assert_called_once_with("Make a plan.")


    def test_draft_plan_is_explicitly_approved(self):
        plan = {
            "id": 7,
            "session_id": 17,
            "status": "draft",
        }

        with (
            mock.patch.object(
                cli.memory,
                "get_plan",
                return_value=plan,
            ),
            mock.patch.object(
                cli.memory,
                "transition_plan",
                return_value=True,
            ) as transition,
            mock.patch.object(
                cli.memory,
                "set_plan_mode",
            ) as set_plan_mode,
        ):
            error = cli._approve_cli_plan(17, 7)

        self.assertIsNone(error)
        transition.assert_called_once_with(
            7,
            "draft",
            "approved",
        )
        set_plan_mode.assert_called_once_with(17, False)

    def test_plan_from_different_session_is_rejected(self):
        plan = {
            "id": 7,
            "session_id": 18,
            "status": "draft",
        }

        with (
            mock.patch.object(
                cli.memory,
                "get_plan",
                return_value=plan,
            ),
            mock.patch.object(
                cli.memory,
                "transition_plan",
            ) as transition,
            mock.patch.object(
                cli.memory,
                "set_plan_mode",
            ) as set_plan_mode,
        ):
            error = cli._approve_cli_plan(17, 7)

        self.assertIn("different thread", error)
        transition.assert_not_called()
        set_plan_mode.assert_not_called()

    def test_run_plan_uses_existing_executor_and_cancel_event(self):
        agent = mock.Mock()
        agent.execute_plan.return_value = "PASS: completed"
        previous_handler = object()

        with (
            mock.patch.object(
                cli.signal,
                "getsignal",
                return_value=previous_handler,
            ),
            mock.patch.object(
                cli.signal,
                "signal",
            ) as set_signal,
            mock.patch.object(
                cli,
                "ensure_visible_reply",
                return_value="PASS: completed",
            ),
            mock.patch("builtins.print"),
        ):
            cli._run_cli_plan(agent, 7)

        agent.execute_plan.assert_called_once()
        call = agent.execute_plan.call_args
        self.assertEqual(call.args, (7,))
        self.assertIn("cancel_event", call.kwargs)
        self.assertFalse(call.kwargs["cancel_event"].is_set())
        self.assertEqual(set_signal.call_count, 2)

    def test_run_plan_main_uses_shared_session_in_normal_mode(self):
        session = {
            "id": 17,
            "title": "Fluxa",
            "folder_path": "/var/www/LiamApp01",
            "plan_mode": True,
        }
        plan = {
            "id": 7,
            "session_id": 17,
            "status": "draft",
        }
        agent = mock.Mock()
        agent.plan_mode = False
        saved = {
            "model": "test-model",
            "auto_confirm": True,
            "custom_instructions": "shared instructions",
        }

        with (
            mock.patch.object(
                cli.sys,
                "argv",
                [
                    "LiamAgent.py",
                    "--session-id",
                    "17",
                    "--run-plan",
                    "7",
                ],
            ),
            mock.patch.object(
                cli.liam_settings,
                "load",
                return_value=saved,
            ),
            mock.patch.object(
                cli.memory,
                "get_session",
                return_value=session,
            ),
            mock.patch.object(
                cli.memory,
                "get_plan",
                return_value=plan,
            ),
            mock.patch.object(
                cli.memory,
                "transition_plan",
                return_value=True,
            ) as transition,
            mock.patch.object(
                cli.memory,
                "set_plan_mode",
            ) as set_plan_mode,
            mock.patch.object(
                cli.memory,
                "list_session_folders",
                return_value=[],
            ),
            mock.patch.object(
                cli,
                "Agent",
                return_value=agent,
            ) as agent_class,
            mock.patch.object(
                cli,
                "_run_cli_plan",
            ) as run_plan,
            mock.patch("builtins.print"),
        ):
            cli.main()

        transition.assert_called_once_with(
            7,
            "draft",
            "approved",
        )
        set_plan_mode.assert_called_once_with(17, False)

        kwargs = agent_class.call_args.kwargs
        self.assertEqual(kwargs["session_id"], 17)
        self.assertEqual(
            kwargs["workdir"],
            "/var/www/LiamApp01",
        )
        self.assertFalse(kwargs["plan_mode"])
        run_plan.assert_called_once_with(agent, 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
