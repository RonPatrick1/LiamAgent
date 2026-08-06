import json
import unittest
from unittest import mock

from agent import memory
from agent.contracts import PersistentActionContractStore


class ActionContractMemoryTests(unittest.TestCase):
    @staticmethod
    def _connection():
        connection = mock.MagicMock()
        cursor = mock.MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        connection.cursor.return_value.__exit__.return_value = False
        return connection, cursor

    def test_create_supersedes_existing_unresolved_contract(self):
        connection, cursor = self._connection()
        cursor.lastrowid = 42
        contract = {
            "operation": "run command",
            "required_capability": "process.execute",
            "completion_mode": "process_exited",
            "preferred_tool": "run_shell_command",
            "targets": {"command": "printf action-ran"},
            "constraints": {},
            "status": "pending",
        }

        with mock.patch.object(
            memory,
            "_connect",
            return_value=connection,
        ), mock.patch.object(memory, "_ensure_schema"):
            contract_id = memory.create_action_contract(
                17,
                "Run printf action-ran.",
                contract,
            )

        self.assertEqual(contract_id, 42)
        connection.begin.assert_called_once()
        connection.commit.assert_called_once()
        connection.rollback.assert_not_called()

        statements = [
            call.args[0]
            for call in cursor.execute.call_args_list
        ]
        self.assertIn(
            "SET status = 'superseded'",
            statements[0],
        )
        self.assertIn(
            "INSERT INTO action_contracts",
            statements[1],
        )

    def test_active_contract_decodes_targets_and_constraints(self):
        connection, cursor = self._connection()
        cursor.fetchone.return_value = (
            42,
            17,
            "Run printf action-ran.",
            "fingerprint",
            "run command",
            "process.execute",
            "process_exited",
            "run_shell_command",
            json.dumps({"command": "printf action-ran"}),
            json.dumps({"sudo": False}),
            "pending",
            None,
            None,
            "created",
            "updated",
            None,
        )

        with mock.patch.object(
            memory,
            "_connect",
            return_value=connection,
        ), mock.patch.object(memory, "_ensure_schema"):
            contract = memory.get_active_action_contract(17)

        self.assertEqual(contract["id"], 42)
        self.assertEqual(
            contract["targets"],
            {"command": "printf action-ran"},
        )
        self.assertEqual(
            contract["constraints"],
            {"sudo": False},
        )

    def test_invalid_transition_is_rejected_before_database_access(self):
        with mock.patch.object(memory, "_connect") as connect:
            with self.assertRaisesRegex(
                ValueError,
                "Invalid action contract transition",
            ):
                memory.transition_action_contract(
                    42,
                    "succeeded",
                    "running",
                )

        connect.assert_not_called()

    def test_record_event_persists_structured_machine_evidence(self):
        connection, cursor = self._connection()
        cursor.lastrowid = 81
        event = {
            "tool": "run_shell_command",
            "status": "success",
            "reason": "completed",
            "capabilities": ["process.execute"],
            "effect_kind": "process_exited",
            "targets": {"command": "printf action-ran"},
            "args": {"command": "printf action-ran"},
            "evidence": {
                "arguments": {
                    "command": "printf action-ran",
                },
            },
            "result": "action-ran\n[exit code: 0]",
        }

        with mock.patch.object(
            memory,
            "_connect",
            return_value=connection,
        ), mock.patch.object(memory, "_ensure_schema"):
            event_id = memory.record_action_tool_event(
                17,
                event,
                contract_id=42,
            )

        self.assertEqual(event_id, 81)
        statement = cursor.execute.call_args.args[0]
        values = cursor.execute.call_args.args[1]
        self.assertIn(
            "INSERT INTO action_tool_events",
            statement,
        )
        self.assertEqual(values[0], 42)
        self.assertEqual(values[1], 17)
        self.assertEqual(values[2], "run_shell_command")
        self.assertEqual(
            json.loads(values[5]),
            ["process.execute"],
        )

    def test_delete_session_removes_contract_evidence_first(self):
        connection, cursor = self._connection()

        with mock.patch.object(
            memory,
            "_connect",
            return_value=connection,
        ), mock.patch.object(memory, "_ensure_schema"):
            result = memory.delete_session(17)

        self.assertEqual(result, "Deleted.")
        statements = [
            call.args[0]
            for call in cursor.execute.call_args_list
        ]
        self.assertIn(
            "DELETE FROM action_tool_events",
            statements[0],
        )
        self.assertIn(
            "DELETE FROM action_contracts",
            statements[1],
        )
        self.assertIn(
            "DELETE FROM sessions",
            statements[-1],
        )

    def test_persistent_store_delegates_to_memory_lifecycle(self):
        store = PersistentActionContractStore()
        contract = {"status": "pending"}

        with mock.patch.object(
            memory,
            "create_action_contract",
            return_value=42,
        ) as create:
            result = store.create(17, "Do it.", contract)

        self.assertEqual(result, 42)
        create.assert_called_once_with(
            17,
            "Do it.",
            contract,
        )


if __name__ == "__main__":
    unittest.main()
