import inspect
import threading
import unittest
from unittest import mock

import LiamGUI
from LiamGUI import LiamWindow


class FakeButton:
    def __init__(self):
        self.visible = None
        self.sensitive = None
        self.label = None

    def set_visible(self, visible):
        self.visible = visible

    def set_sensitive(self, sensitive):
        self.sensitive = sensitive

    def set_label(self, label):
        self.label = label


class GuiPlanExecutionTests(unittest.TestCase):
    def test_header_contains_run_and_stop_controls(self):
        source = inspect.getsource(LiamWindow.__init__)

        self.assertIn(
            'self.execute_plan_button = Gtk.Button(label="Run Plan")',
            source,
        )
        self.assertIn(
            'self.cancel_plan_button = Gtk.Button(label="Stop")',
            source,
        )
        self.assertIn(
            "self._on_execute_plan_clicked",
            source,
        )
        self.assertIn(
            "self._on_cancel_plan_clicked",
            source,
        )

    @mock.patch.object(LiamGUI.memory, "get_latest_plan")
    def test_draft_plan_enables_run_button(
        self,
        get_latest_plan,
    ):
        get_latest_plan.return_value = {
            "id": 41,
            "status": "draft",
        }

        window = LiamWindow.__new__(LiamWindow)
        window.session_id = 17
        window.busy = False
        window._executing_plan_id = None
        window._plan_cancel_event = None
        window.execute_plan_button = FakeButton()
        window.cancel_plan_button = FakeButton()

        window._refresh_plan_actions()

        self.assertEqual(
            window.execute_plan_button.label,
            "Run Plan",
        )
        self.assertTrue(
            window.execute_plan_button.visible
        )
        self.assertTrue(
            window.execute_plan_button.sensitive
        )
        self.assertFalse(
            window.cancel_plan_button.visible
        )

    @mock.patch.object(LiamGUI.threading, "Thread")
    @mock.patch.object(LiamGUI.memory, "set_plan_mode")
    @mock.patch.object(
        LiamGUI.memory,
        "transition_plan",
        return_value=True,
    )
    @mock.patch.object(LiamGUI.memory, "get_plan")
    def test_start_approves_draft_and_launches_normal_agent(
        self,
        get_plan,
        transition_plan,
        set_plan_mode,
        thread_class,
    ):
        get_plan.return_value = {
            "id": 41,
            "session_id": 17,
            "status": "draft",
        }

        normal_agent = mock.Mock()
        normal_agent.plan_mode = False

        window = LiamWindow.__new__(LiamWindow)
        window.busy = False
        window.session_id = 17
        window.agent = normal_agent
        window._executing_plan_id = None
        window._plan_cancel_event = None
        window._reload_current_agent = mock.Mock()
        window._set_busy = mock.Mock()
        window._start_thinking = mock.Mock()
        window._refresh_plan_actions = mock.Mock()
        window._append_message = mock.Mock()

        window._start_approved_plan(41)

        transition_plan.assert_called_once_with(
            41,
            "draft",
            "approved",
        )
        set_plan_mode.assert_called_once_with(17, False)
        window._reload_current_agent.assert_called_once_with()
        self.assertEqual(window._executing_plan_id, 41)
        self.assertIsInstance(
            window._plan_cancel_event,
            threading.Event,
        )
        window._set_busy.assert_called_once_with(True)
        window._start_thinking.assert_called_once_with()
        thread_class.assert_called_once_with(
            target=window._run_approved_plan,
            args=(
                normal_agent,
                41,
                window._plan_cancel_event,
            ),
            daemon=True,
        )
        thread_class.return_value.start.assert_called_once_with()

    def test_cancel_sets_execution_event(self):
        window = LiamWindow.__new__(LiamWindow)
        window._plan_cancel_event = threading.Event()
        window.cancel_plan_button = FakeButton()
        window._append_message = mock.Mock()

        window._on_cancel_plan_clicked(None)

        self.assertTrue(
            window._plan_cancel_event.is_set()
        )
        self.assertFalse(
            window.cancel_plan_button.sensitive
        )
        window._append_message.assert_called_once()
        self.assertIn(
            "next safe checkpoint",
            window._append_message.call_args.args[0],
        )

    def test_worker_calls_executor_and_schedules_finish(self):
        agent = mock.Mock()
        agent.execute_plan.return_value = "PASS: complete"
        cancel_event = threading.Event()

        window = LiamWindow.__new__(LiamWindow)

        with mock.patch.object(
            LiamGUI.GLib,
            "idle_add",
        ) as idle_add:
            window._run_approved_plan(
                agent,
                41,
                cancel_event,
            )

        agent.execute_plan.assert_called_once_with(
            41,
            cancel_event=cancel_event,
        )
        idle_add.assert_called_once_with(
            window._finish_plan_execution,
            "PASS: complete",
            41,
            cancel_event,
        )

    def test_finish_clears_execution_state(self):
        cancel_event = threading.Event()

        window = LiamWindow.__new__(LiamWindow)
        window._executing_plan_id = 41
        window._plan_cancel_event = cancel_event
        window._stop_thinking = mock.Mock()
        window._append_message = mock.Mock()
        window._set_busy = mock.Mock()
        window._refresh_plan_actions = mock.Mock()

        result = window._finish_plan_execution(
            "PASS: complete",
            41,
            cancel_event,
        )

        self.assertFalse(result)
        self.assertIsNone(window._executing_plan_id)
        self.assertIsNone(window._plan_cancel_event)
        window._stop_thinking.assert_called_once_with()
        window._append_message.assert_called_once_with(
            "PASS: complete",
            "assistant",
            use_markup=True,
        )
        window._set_busy.assert_called_once_with(False)
        window._refresh_plan_actions.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
