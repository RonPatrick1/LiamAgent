import unittest
from unittest import mock

from agent import core


class FakeContractStore:
    def __init__(self, active=None):
        self.active = active
        self.events = []
        self.transitions = []

    def get_active(self, session_id):
        return self.active

    def record_event(self, session_id, event, contract_id=None):
        self.events.append({
            "session_id": session_id,
            "event": dict(event),
            "contract_id": contract_id,
        })
        return 81

    def transition(
        self,
        contract_id,
        expected_status,
        new_status,
        matched_event_id=None,
        failure_reason=None,
    ):
        self.transitions.append({
            "contract_id": contract_id,
            "expected_status": expected_status,
            "new_status": new_status,
            "matched_event_id": matched_event_id,
            "failure_reason": failure_reason,
        })
        return True


class AgentContractIntegrationTests(unittest.TestCase):
    @staticmethod
    def _agent(contract=None, store=None):
        agent = core.Agent.__new__(core.Agent)
        agent.session_id = 17
        agent.action_contract_store = store
        agent._active_action_contract = contract
        agent._tool_events = []
        agent.on_status = mock.Mock()
        return agent

    @staticmethod
    def _shell_event(command="printf action-ran", status="success"):
        return {
            "tool": "run_shell_command",
            "args": {"command": command},
            "result": "action-ran\n[exit code: 0]",
            "status": status,
            "reason": "completed" if status == "success" else "nonzero_exit",
            "capabilities": ["process.execute"],
            "effect_kind": "process_exited",
            "completion_modes": ["process_exited"],
            "targets": {"command": command},
            "evidence": {
                "arguments": {
                    "command": command,
                },
            },
        }

    def test_matching_event_persists_and_completes_contract(self):
        contract = {
            "id": 42,
            "status": "pending",
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": "run_shell_command",
            "targets": {"command": "printf action-ran"},
            "constraints": {},
        }
        store = FakeContractStore(active=contract)
        agent = self._agent(contract=contract, store=store)
        event = self._shell_event()

        agent._persist_and_match_action_event(event)

        self.assertEqual(len(store.events), 1)
        self.assertEqual(store.events[0]["contract_id"], 42)
        self.assertEqual(
            store.transitions,
            [{
                "contract_id": 42,
                "expected_status": "pending",
                "new_status": "succeeded",
                "matched_event_id": 81,
                "failure_reason": None,
            }],
        )
        self.assertTrue(event["contract_match"])
        self.assertEqual(
            agent._active_action_contract["status"],
            "succeeded",
        )
        self.assertEqual(
            agent._active_action_contract["matched_event_id"],
            81,
        )

    def test_wrong_successful_tool_event_does_not_complete_contract(self):
        contract = {
            "id": 42,
            "status": "pending",
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": "run_shell_command",
            "targets": {"command": "printf action-ran"},
            "constraints": {},
        }
        store = FakeContractStore(active=contract)
        agent = self._agent(contract=contract, store=store)
        event = {
            "tool": "read_file",
            "args": {"path": "/media/example/Ride.flac"},
            "result": "binary data",
            "status": "success",
            "reason": "completed",
            "capabilities": ["filesystem.read"],
            "effect_kind": "state_observed",
            "completion_modes": ["state_observed"],
            "targets": {"path": "/media/example/Ride.flac"},
            "evidence": {
                "arguments": {
                    "path": "/media/example/Ride.flac",
                },
            },
        }

        agent._persist_and_match_action_event(event)

        self.assertEqual(len(store.events), 1)
        self.assertEqual(store.transitions, [])
        self.assertFalse(event["contract_match"])
        self.assertEqual(
            agent._active_action_contract["status"],
            "pending",
        )

    def test_failed_matching_tool_does_not_complete_contract(self):
        contract = {
            "id": 42,
            "status": "pending",
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": "run_shell_command",
            "targets": {"command": "printf action-ran"},
            "constraints": {},
        }
        store = FakeContractStore(active=contract)
        agent = self._agent(contract=contract, store=store)
        event = self._shell_event(status="failure")

        agent._persist_and_match_action_event(event)

        self.assertEqual(store.transitions, [])
        self.assertFalse(event["contract_match"])
        self.assertEqual(
            agent._active_action_contract["status"],
            "pending",
        )

    def test_store_failure_never_marks_contract_complete(self):
        contract = {
            "id": 42,
            "status": "pending",
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": "run_shell_command",
            "targets": {"command": "printf action-ran"},
            "constraints": {},
        }
        store = mock.Mock()
        store.record_event.side_effect = RuntimeError("database unavailable")
        agent = self._agent(contract=contract, store=store)
        event = self._shell_event()

        agent._persist_and_match_action_event(event)

        store.transition.assert_not_called()
        self.assertNotIn("contract_match", event)
        self.assertEqual(
            agent._active_action_contract["status"],
            "pending",
        )
        agent.on_status.assert_called_once()

    def test_no_store_preserves_existing_unit_test_behavior(self):
        agent = self._agent(contract=None, store=None)
        event = self._shell_event()

        agent._persist_and_match_action_event(event)

        self.assertNotIn("persistent_event_id", event)
        self.assertNotIn("contract_match", event)


if __name__ == "__main__":
    unittest.main()
