import unittest

from agent import core
from agent.contracts import (
    build_tool_event,
    event_satisfies_contract,
    validate_contract_proposal,
)
from agent.tools import TOOL_DEFINITIONS, TOOL_IMPL


class ToolDefinitionTests(unittest.TestCase):
    def test_every_real_tool_has_exactly_one_definition(self):
        self.assertEqual(set(TOOL_DEFINITIONS), set(TOOL_IMPL))

    def test_definitions_have_required_authoritative_metadata(self):
        for name, definition in TOOL_DEFINITIONS.items():
            with self.subTest(tool=name):
                self.assertTrue(definition["capabilities"])
                self.assertTrue(definition["effect_kind"])
                self.assertIn(
                    definition["effect_kind"],
                    definition["completion_modes"],
                )
                self.assertIsInstance(definition["target_fields"], tuple)


class ContractValidationTests(unittest.TestCase):
    def test_non_action_returns_none(self):
        proposal = {
            "actionable": False,
            "operation": None,
            "required_capability": None,
            "completion_mode": None,
            "preferred_tool": None,
            "targets": {},
            "constraints": {},
            "needs_clarification": False,
        }

        self.assertIsNone(
            validate_contract_proposal(proposal, TOOL_DEFINITIONS)
        )

    def test_valid_shell_contract_is_host_validated(self):
        proposal = {
            "actionable": True,
            "operation": "run command",
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": "run_shell_command",
            "targets": {"command": "printf action-ran"},
            "constraints": {},
            "needs_clarification": False,
        }

        contract = validate_contract_proposal(
            proposal,
            TOOL_DEFINITIONS,
            offered_tool_names={"run_shell_command", "read_file"},
        )

        self.assertEqual(contract["status"], "pending")
        self.assertEqual(
            contract["required_capability"],
            "process.execute",
        )

    def test_unoffered_preferred_tool_is_rejected(self):
        proposal = {
            "actionable": True,
            "operation": "run command",
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": "run_shell_command",
            "targets": {"command": "printf action-ran"},
            "constraints": {},
            "needs_clarification": False,
        }

        with self.assertRaisesRegex(ValueError, "not offered"):
            validate_contract_proposal(
                proposal,
                TOOL_DEFINITIONS,
                offered_tool_names={"read_file"},
            )

    def test_model_cannot_assign_wrong_capability_to_tool(self):
        proposal = {
            "actionable": True,
            "operation": "run command",
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": "read_file",
            "targets": {"path": "/tmp/file"},
            "constraints": {},
            "needs_clarification": False,
        }

        with self.assertRaisesRegex(ValueError, "does not provide"):
            validate_contract_proposal(
                proposal,
                TOOL_DEFINITIONS,
            )


class ContractMatchingTests(unittest.TestCase):
    def test_read_event_cannot_complete_process_contract(self):
        contract = {
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": None,
            "targets": {},
            "constraints": {},
        }
        event = build_tool_event(
            "read_file",
            {"path": "/media/example/Ride.flac"},
            "binary data",
            {
                "tool": "read_file",
                "status": "success",
                "reason": "completed",
            },
            TOOL_DEFINITIONS,
        )

        self.assertFalse(event_satisfies_contract(contract, event))

    def test_exact_successful_shell_event_completes_contract(self):
        contract = {
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": "run_shell_command",
            "targets": {"command": "printf action-ran"},
            "constraints": {},
        }
        event = build_tool_event(
            "run_shell_command",
            {"command": "printf action-ran"},
            "action-ran\n[exit code: 0]",
            {
                "tool": "run_shell_command",
                "status": "success",
                "reason": "completed",
            },
            TOOL_DEFINITIONS,
        )

        self.assertTrue(event_satisfies_contract(contract, event))

    def test_failed_event_never_completes_contract(self):
        contract = {
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": "run_shell_command",
            "targets": {"command": "false"},
            "constraints": {},
        }
        event = build_tool_event(
            "run_shell_command",
            {"command": "false"},
            "[exit code: 1]",
            {
                "tool": "run_shell_command",
                "status": "failure",
                "reason": "nonzero_exit",
            },
            TOOL_DEFINITIONS,
        )

        self.assertFalse(event_satisfies_contract(contract, event))

    def test_target_mismatch_keeps_contract_pending(self):
        contract = {
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": "run_shell_command",
            "targets": {"command": "printf requested"},
            "constraints": {},
        }
        event = build_tool_event(
            "run_shell_command",
            {"command": "printf something-else"},
            "[exit code: 0]",
            {
                "tool": "run_shell_command",
                "status": "success",
                "reason": "completed",
            },
            TOOL_DEFINITIONS,
        )

        self.assertFalse(event_satisfies_contract(contract, event))

    def test_core_classifier_now_emits_structured_event_metadata(self):
        agent = core.Agent.__new__(core.Agent)

        event = agent._classify_tool_outcome(
            "run_shell_command",
            {"command": "printf action-ran"},
            "action-ran\n[exit code: 0]",
        )

        self.assertEqual(event["status"], "success")
        self.assertIn("process.execute", event["capabilities"])
        self.assertEqual(event["effect_kind"], "process_exited")
        self.assertEqual(
            event["targets"],
            {"command": "printf action-ran"},
        )


if __name__ == "__main__":
    unittest.main()
