import unittest
from pathlib import Path
from unittest import mock

from agent import memory


class PlanExecutionMemoryTests(unittest.TestCase):
    def setUp(self):
        self.connection = mock.MagicMock()
        self.cursor = mock.MagicMock()
        self.connection.cursor.return_value.__enter__.return_value = self.cursor

        connect_patch = mock.patch.object(
            memory,
            "_connect",
            return_value=self.connection,
        )
        schema_patch = mock.patch.object(memory, "_ensure_schema")

        connect_patch.start()
        schema_patch.start()

        self.addCleanup(connect_patch.stop)
        self.addCleanup(schema_patch.stop)

    def test_schema_contains_dedicated_plan_lifecycle_table(self):
        source = Path(memory.__file__).read_text()

        self.assertIn("CREATE TABLE IF NOT EXISTS plans", source)
        self.assertIn(
            "status VARCHAR(16) NOT NULL DEFAULT 'draft'",
            source,
        )
        self.assertIn("approved_at TIMESTAMP NULL", source)
        self.assertIn("started_at TIMESTAMP NULL", source)
        self.assertIn("completed_at TIMESTAMP NULL", source)

    def test_create_plan_stores_unapproved_draft(self):
        self.cursor.lastrowid = 41

        plan_id = memory.create_plan(17, "  Inspect, modify, validate.  ")

        self.assertEqual(plan_id, 41)
        self.cursor.execute.assert_called_once_with(
            "INSERT INTO plans (session_id, content, status) "
            "VALUES (%s, %s, 'draft')",
            (17, "Inspect, modify, validate."),
        )

    def test_latest_plan_returns_full_lifecycle_record(self):
        row = (
            41,
            17,
            "Inspect, modify, validate.",
            "approved",
            None,
            "created",
            "updated",
            "approved",
            None,
            None,
        )
        self.cursor.fetchone.return_value = row

        plan = memory.get_latest_plan(17)

        self.assertEqual(plan["id"], 41)
        self.assertEqual(plan["session_id"], 17)
        self.assertEqual(plan["status"], "approved")
        self.assertEqual(plan["content"], "Inspect, modify, validate.")
        sql, params = self.cursor.execute.call_args.args
        self.assertIn("FROM plans WHERE session_id = %s", sql)
        self.assertIn("ORDER BY id DESC LIMIT 1", sql)
        self.assertEqual(params, (17,))

    def test_approval_is_an_atomic_expected_state_transition(self):
        self.cursor.rowcount = 1

        changed = memory.transition_plan(41, "draft", "approved")

        self.assertTrue(changed)
        sql, params = self.cursor.execute.call_args.args
        self.assertIn("status = %s", sql)
        self.assertIn("approved_at = CURRENT_TIMESTAMP", sql)
        self.assertIn("WHERE id = %s AND status = %s", sql)
        self.assertEqual(params, ("approved", 41, "draft"))

    def test_completion_records_result_and_timestamp(self):
        self.cursor.rowcount = 1

        changed = memory.transition_plan(
            41,
            "running",
            "passed",
            result="PASS: all acceptance tests passed",
        )

        self.assertTrue(changed)
        sql, params = self.cursor.execute.call_args.args
        self.assertIn("result = %s", sql)
        self.assertIn("completed_at = CURRENT_TIMESTAMP", sql)
        self.assertEqual(
            params,
            (
                "passed",
                "PASS: all acceptance tests passed",
                41,
                "running",
            ),
        )

    def test_invalid_transition_is_rejected_before_database_access(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid plan transition",
        ):
            memory.transition_plan(41, "draft", "running")

        memory._connect.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
