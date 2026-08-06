import json
import unittest
from unittest import mock

from agent import core
from agent.tools import TOOL_SCHEMAS


class FakeStore:
    def __init__(self):
        self.created = []

    def create(self, session_id, source_text, contract):
        self.created.append((session_id, source_text, contract))
        return 42

    def get(self, contract_id):
        session_id, source_text, contract = self.created[-1]
        stored = dict(contract)
        stored.update({
            "id": contract_id,
            "session_id": session_id,
            "source_text": source_text,
        })
        return stored


class ActionContractExtractionTests(unittest.TestCase):
    @staticmethod
    def _agent(store=None):
        agent = core.Agent.__new__(core.Agent)
        agent.session_id = 17
        agent.action_contract_store = store
        agent._active_action_contract = None
        agent.plan_mode = False
        agent._turn_plan_mode = False
        agent.on_status = mock.Mock()
        agent.tool_schemas = [
            schema
            for schema in TOOL_SCHEMAS
            if schema["function"]["name"] in {
                "read_file",
                "run_shell_command",
                "start_process",
            }
        ]
        return agent

    def test_valid_proposal_is_host_validated_and_persisted(self):
        store = FakeStore()
        agent = self._agent(store)
        agent._helper_chat = mock.Mock(return_value={
            "content": json.dumps({
                "actionable": True,
                "operation": "play media",
                "required_capability": "process.start",
                "completion_mode": "process_started",
                "preferred_tool": "start_process",
                "targets": {
                    "argv": [
                        "ogg123",
                        "/media/example/Ride.flac",
                    ],
                },
                "constraints": {},
                "needs_clarification": False,
            }),
        })

        contract = agent._prepare_action_contract(
            "Use ogg123 to play /media/example/Ride.flac"
        )

        self.assertEqual(contract["id"], 42)
        self.assertEqual(contract["status"], "pending")
        self.assertEqual(
            contract["targets"]["argv"],
            ["ogg123", "/media/example/Ride.flac"],
        )
        self.assertEqual(len(store.created), 1)
        self.assertIs(
            agent._helper_chat.call_args.kwargs["response_format"],
            core.ACTION_CONTRACT_PROPOSAL_SCHEMA,
        )

    def test_pending_contract_survives_followup_without_reclassification(self):
        existing = {
            "id": 42,
            "status": "pending",
            "operation": "play media",
            "required_capability": "process.start",
            "completion_mode": "process_started",
            "preferred_tool": "start_process",
            "targets": {
                "argv": [
                    "ogg123",
                    "/media/example/Ride.flac",
                ],
            },
            "constraints": {},
        }
        agent = self._agent(FakeStore())
        agent._active_action_contract = existing
        agent._helper_chat = mock.Mock()

        result = agent._prepare_action_contract("do that then")

        self.assertIs(result, existing)
        agent._helper_chat.assert_not_called()

    def test_invalid_json_creates_no_contract(self):
        store = FakeStore()
        agent = self._agent(store)
        agent._helper_chat = mock.Mock(return_value={
            "content": "not json",
        })

        result = agent._prepare_action_contract("run it")

        self.assertIsNone(result)
        self.assertEqual(store.created, [])
        agent.on_status.assert_called_once()

    def test_unresolved_contract_adds_authoritative_notice(self):
        agent = self._agent()
        agent._active_action_contract = {
            "id": 42,
            "status": "pending",
            "operation": "play media",
        }

        result = agent._enforce_action_contract_status(
            "The music is playing."
        )

        self.assertIn("The music is playing.", result)
        self.assertIn(
            "no matching successful tool event completed play media",
            result,
        )

    def test_succeeded_contract_adds_no_notice(self):
        agent = self._agent()
        agent._active_action_contract = {
            "id": 42,
            "status": "succeeded",
            "operation": "play media",
        }

        self.assertEqual(
            agent._enforce_action_contract_status(
                "The music is playing."
            ),
            "The music is playing.",
        )

    def test_contract_instruction_preserves_exact_target(self):
        instruction = core.Agent._action_contract_instruction({
            "id": 42,
            "status": "pending",
            "operation": "play media",
            "required_capability": "process.start",
            "completion_mode": "process_started",
            "preferred_tool": "start_process",
            "targets": {
                "argv": [
                    "ogg123",
                    "/media/example/Ride.flac",
                ],
            },
        })

        self.assertIn(
            "AUTHORITATIVE HOST ACTION CONTRACT",
            instruction,
        )
        self.assertIn("/media/example/Ride.flac", instruction)
        self.assertIn("start_process", instruction)


if __name__ == "__main__":
    unittest.main()
