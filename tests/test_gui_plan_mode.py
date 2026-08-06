import inspect
import unittest
from unittest import mock

import LiamGUI
from LiamGUI import LiamWindow


class FakeToggle:
    def __init__(self, active=False):
        self.active = active
        self.sensitive = None

    def get_active(self):
        return self.active

    def set_sensitive(self, sensitive):
        self.sensitive = sensitive


class GuiPlanModeTests(unittest.TestCase):
    def test_header_contains_plan_toggle_and_tooltip(self):
        source = inspect.getsource(LiamWindow.__init__)

        self.assertIn(
            'self.plan_toggle = Gtk.ToggleButton(label="Plan")',
            source,
        )
        self.assertIn(
            'self.plan_toggle.set_tooltip_text("Plan without making changes")',
            source,
        )
        self.assertIn(
            'self.plan_toggle.connect("toggled", self._on_plan_toggled)',
            source,
        )

    @mock.patch.object(LiamGUI.memory, "set_plan_mode")
    def test_toggle_persists_mode_and_rebuilds_agent(self, set_plan_mode):
        window = LiamWindow.__new__(LiamWindow)
        window._setting_plan_toggle = False
        window.session_id = 17
        window._reload_current_agent = mock.Mock()
        button = FakeToggle(active=True)

        window._on_plan_toggled(button)

        set_plan_mode.assert_called_once_with(17, True)
        window._reload_current_agent.assert_called_once_with()

    @mock.patch.object(LiamGUI.memory, "set_plan_mode")
    def test_programmatic_toggle_sync_does_not_persist_again(self, set_plan_mode):
        window = LiamWindow.__new__(LiamWindow)
        window._setting_plan_toggle = True
        window.session_id = 17
        window._reload_current_agent = mock.Mock()

        window._on_plan_toggled(FakeToggle(active=True))

        set_plan_mode.assert_not_called()
        window._reload_current_agent.assert_not_called()

    @mock.patch.object(LiamGUI.memory, "set_plan_mode")
    def test_toggle_without_active_session_does_nothing(self, set_plan_mode):
        window = LiamWindow.__new__(LiamWindow)
        window._setting_plan_toggle = False
        window.session_id = None
        window._reload_current_agent = mock.Mock()

        window._on_plan_toggled(FakeToggle(active=True))

        set_plan_mode.assert_not_called()
        window._reload_current_agent.assert_not_called()

    @mock.patch.object(LiamGUI.memory, "load_recent_messages", return_value=[])
    @mock.patch.object(LiamGUI.memory, "list_session_folders", return_value=[])
    @mock.patch.object(
        LiamGUI.memory,
        "get_session",
        return_value={"id": 17, "plan_mode": True},
    )
    @mock.patch.object(LiamGUI, "Agent")
    def test_agent_build_uses_persisted_thread_mode(
        self,
        agent_class,
        get_session,
        list_session_folders,
        load_recent_messages,
    ):
        fake_agent = mock.Mock()
        agent_class.return_value = fake_agent

        window = LiamWindow.__new__(LiamWindow)
        window.model = "test-model"
        window.auto_confirm = False
        window.settings = {"custom_instructions": "test instructions"}

        agent, history = window._build_agent_and_history(
            17,
            "/var/www/LiamAgent",
        )

        self.assertIs(agent, fake_agent)
        self.assertEqual(history, [])
        get_session.assert_called_once_with(17)
        list_session_folders.assert_called_once_with(17)
        load_recent_messages.assert_called_once_with(
            limit=LiamGUI.REPLAY_LIMIT,
            session_id=17,
        )
        agent_class.assert_called_once_with(
            model="test-model",
            auto_confirm=False,
            workdir="/var/www/LiamAgent",
            session_id=17,
            extra_folders=[],
            custom_instructions="test instructions",
            channel="gui",
            actor_id="local-owner",
            is_owner=True,
            learning_enabled=True,
            plan_mode=True,
            sudo_enabled=False,
            action_contract_store=mock.ANY,
        )

    def test_busy_state_disables_plan_toggle(self):
        window = LiamWindow.__new__(LiamWindow)
        window.entry = mock.Mock()
        window.send_button = mock.Mock()
        window.session_list = mock.Mock()
        window.external_sessions_toggle = mock.Mock()
        window.plan_toggle = FakeToggle()

        window._set_busy(True)

        self.assertTrue(window.busy)
        self.assertFalse(window.plan_toggle.sensitive)


if __name__ == "__main__":
    unittest.main(verbosity=2)
