import os
import tempfile
import unittest
from unittest import mock

from agent import routines


class RoutineCalendarTests(unittest.TestCase):
    def test_once_daily_and_hourly_calendars_are_validated(self):
        self.assertEqual(
            routines._on_calendar("once", "2099-07-29 11:30:00"),
            "2099-07-29 11:30:00",
        )
        self.assertEqual(
            routines._on_calendar("daily", "08:05"),
            "*-*-* 08:05:00",
        )
        self.assertEqual(
            routines._on_calendar("hourly", "4"),
            "*-*-* 0/4:00:00",
        )

    def test_past_once_and_invalid_hourly_interval_are_rejected(self):
        with self.assertRaises(ValueError):
            routines._on_calendar("once", "2000-01-01 00:00:00")
        with self.assertRaises(ValueError):
            routines._on_calendar("hourly", "0")

    def test_unit_creation_checks_systemd_success(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             mock.patch.object(routines, "SYSTEMD_USER_DIR", temp_dir), \
             mock.patch.object(routines.subprocess, "run") as run:
            routines._write_units(42, "once", "2099-07-29 11:30:00")

            with open(os.path.join(temp_dir, "liam-routine-42.timer")) as file:
                timer = file.read()
            self.assertIn("OnCalendar=2099-07-29 11:30:00", timer)
            self.assertIn("AccuracySec=1s", timer)
            self.assertEqual(run.call_count, 3)
            self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
