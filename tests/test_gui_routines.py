from datetime import datetime
import unittest

from LiamGUI import _finished_once_status, _routine_display_prompt


class RoutineDisplayTests(unittest.TestCase):
    def test_execution_wrapper_is_hidden_from_routine_preview(self):
        self.assertEqual(
            _routine_display_prompt(
                "This is the scheduled execution time.\n\nOriginal request:\nSend the reminder"
            ),
            "Send the reminder",
        )
        self.assertEqual(
            _routine_display_prompt("Return exactly this message: Call the dentist"),
            "Call the dentist",
        )

    def test_completed_once_routine_has_terminal_status(self):
        routine = {
            "schedule_kind": "once",
            "schedule_value": "2099-07-29 11:30:00",
            "last_run_at": "2026-07-29 10:15:00",
        }
        self.assertEqual(_finished_once_status(routine), "Completed")

    def test_past_once_routine_is_expired_but_future_once_is_toggleable(self):
        now = datetime(2026, 7, 29, 12, 0, 0)
        expired = {
            "schedule_kind": "once",
            "schedule_value": "2026-07-29 11:30:00",
            "last_run_at": None,
        }
        future = {
            "schedule_kind": "once",
            "schedule_value": "2026-07-29 12:30:00",
            "last_run_at": None,
        }
        recurring = {
            "schedule_kind": "daily",
            "schedule_value": "12:30",
            "last_run_at": None,
        }

        self.assertEqual(_finished_once_status(expired, now), "Expired")
        self.assertIsNone(_finished_once_status(future, now))
        self.assertIsNone(_finished_once_status(recurring, now))


if __name__ == "__main__":
    unittest.main()
