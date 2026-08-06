import unittest
from pathlib import Path
from unittest import mock

from agent import memory


class PlanModeMemoryTests(unittest.TestCase):
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

    def test_schema_contains_idempotent_plan_mode_upgrade(self):
        source = Path(memory.__file__).read_text()

        self.assertIn(
            "SHOW COLUMNS FROM sessions LIKE 'plan_mode'",
            source,
        )
        self.assertIn(
            "ALTER TABLE sessions ADD COLUMN plan_mode",
            source,
        )
        self.assertIn(
            "plan_mode TINYINT(1) NOT NULL DEFAULT 0",
            source,
        )

    def test_get_session_returns_plan_mode(self):
        self.cursor.fetchone.return_value = (
            7,
            "LiamAgent",
            "/var/www/LiamAgent",
            1,
            0,
            0,
            1,
            0,
        )

        result = memory.get_session(7)

        self.assertEqual(
            result,
            {
                "id": 7,
                "title": "LiamAgent",
                "folder_path": "/var/www/LiamAgent",
                "pinned": True,
                "unread": False,
                "archived": False,
                "plan_mode": True,
                "sudo_enabled": False,
            },
        )
        self.cursor.execute.assert_called_once_with(
            "SELECT id, title, folder_path, pinned, unread, archived, "
            "plan_mode, sudo_enabled "
            "FROM sessions WHERE id = %s",
            (7,),
        )

    def test_list_sessions_returns_independent_plan_mode_values(self):
        self.cursor.fetchall.return_value = [
            (
                7,
                "Normal",
                "/tmp/normal",
                "2026-07-31 12:00:00",
                0,
                0,
                0,
                0,
                None,
                None,
            ),
            (
                8,
                "Planning",
                "/tmp/planning",
                "2026-07-31 12:01:00",
                0,
                0,
                0,
                1,
                3,
                "Work",
            ),
        ]

        result = memory.list_sessions()

        self.assertFalse(result[0]["plan_mode"])
        self.assertTrue(result[1]["plan_mode"])
        self.assertEqual(result[1]["group_id"], 3)
        self.assertEqual(result[1]["group_name"], "Work")

    def test_set_plan_mode_writes_boolean_as_tinyint(self):
        memory.set_plan_mode(7, True)

        self.cursor.execute.assert_called_once_with(
            "UPDATE sessions SET plan_mode = %s WHERE id = %s",
            (1, 7),
        )

        self.cursor.execute.reset_mock()

        memory.set_plan_mode(7, False)

        self.cursor.execute.assert_called_once_with(
            "UPDATE sessions SET plan_mode = %s WHERE id = %s",
            (0, 7),
        )

    def test_fork_session_copies_plan_mode(self):
        self.cursor.fetchone.return_value = (
            "LiamAgent",
            "/var/www/LiamAgent",
            4,
            1,
        )
        self.cursor.lastrowid = 22

        new_id = memory.fork_session(7)

        self.assertEqual(new_id, 22)

        calls = self.cursor.execute.call_args_list
        self.assertEqual(
            calls[0],
            mock.call(
                "SELECT title, folder_path, group_id, plan_mode "
                "FROM sessions WHERE id = %s",
                (7,),
            ),
        )
        self.assertEqual(
            calls[1],
            mock.call(
                "INSERT INTO sessions "
                "(title, folder_path, group_id, plan_mode) "
                "VALUES (%s, %s, %s, %s)",
                (
                    "LiamAgent (fork)",
                    "/var/www/LiamAgent",
                    4,
                    1,
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
