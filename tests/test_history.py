import inspect
import unittest
from unittest import mock

from agent import memory
from LiamGUI import (
    HISTORY_HELP,
    LiamWindow,
    _CommandHistoryNavigator,
    _format_command_history,
    _parse_history_request,
)


class CommandHistoryStorageTests(unittest.TestCase):
    def _connection(self):
        connection = mock.MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        return connection, cursor

    def test_schema_contains_dedicated_command_history_table(self):
        connection, cursor = self._connection()

        memory._ensure_schema(connection)

        statements = "\n".join(
            call.args[0] for call in cursor.execute.call_args_list if call.args
        )
        self.assertIn("CREATE TABLE IF NOT EXISTS command_history", statements)
        self.assertIn("command_text MEDIUMTEXT NOT NULL", statements)

    def test_save_inserts_and_prunes_older_than_the_newest_thousand(self):
        connection, cursor = self._connection()
        cursor.fetchone.return_value = (41,)

        with mock.patch.object(memory, "_connect", return_value=connection), \
             mock.patch.object(memory, "_ensure_schema"):
            saved = memory.save_command_history("ollama list", session_id=7)

        self.assertTrue(saved)
        calls = cursor.execute.call_args_list
        self.assertIn("INSERT INTO command_history", calls[0].args[0])
        self.assertEqual(calls[0].args[1], (7, "ollama list"))
        self.assertIn("LIMIT 1 OFFSET", calls[1].args[0])
        self.assertEqual(calls[1].args[1], (memory.COMMAND_HISTORY_LIMIT - 1,))
        self.assertIn("DELETE FROM command_history WHERE id <", calls[2].args[0])
        self.assertEqual(calls[2].args[1], (41,))
        connection.close.assert_called_once()

    def test_load_returns_oldest_first_and_caps_requested_limit(self):
        connection, cursor = self._connection()
        cursor.fetchall.return_value = [
            (3, 8, "third", "later"),
            (2, 7, "second", "earlier"),
        ]

        with mock.patch.object(memory, "_connect", return_value=connection), \
             mock.patch.object(memory, "_ensure_schema"):
            rows = memory.load_command_history(5000)

        self.assertEqual([row["id"] for row in rows], [2, 3])
        self.assertEqual([row["command_text"] for row in rows], ["second", "third"])
        self.assertEqual(
            cursor.execute.call_args.args[1], (memory.COMMAND_HISTORY_LIMIT,),
        )


class CommandHistoryNavigatorTests(unittest.TestCase):
    def test_previous_next_and_draft_restoration_match_shell_behavior(self):
        history = _CommandHistoryNavigator(["one", "two", "three"])

        self.assertEqual(history.previous("unfinished"), "three")
        self.assertEqual(history.previous("three"), "two")
        self.assertEqual(history.next("two"), "three")
        self.assertEqual(history.next("three"), "unfinished")
        self.assertIsNone(history.next("unfinished"))

    def test_oldest_newest_and_hard_entry_cap(self):
        history = _CommandHistoryNavigator(["one", "two"], limit=3)
        history.record("three")
        history.record("four")

        self.assertEqual(history.entries, ["two", "three", "four"])
        self.assertEqual(history.oldest("draft"), "two")
        self.assertEqual(history.newest("two"), "draft")

    def test_prefix_navigation_keeps_the_original_prefix(self):
        history = _CommandHistoryNavigator([
            "docker ps", "ollama list", "docker images", "docker system df",
        ])

        self.assertEqual(history.prefix("docker", -1), "docker system df")
        self.assertEqual(history.prefix("docker system df", -1), "docker images")
        self.assertEqual(history.prefix("docker images", 1), "docker system df")

    def test_incremental_search_is_case_insensitive_and_bidirectional(self):
        history = _CommandHistoryNavigator([
            "systemctl status ollama", "docker ps", "OLLAMA LIST",
        ])

        newest = history.search("ollama", -1)
        self.assertEqual(newest, (2, "OLLAMA LIST"))
        self.assertEqual(
            history.search("ollama", -1, start=newest[0]),
            (0, "systemctl status ollama"),
        )
        self.assertEqual(
            history.search("ollama", 1, start=0),
            (2, "OLLAMA LIST"),
        )


class _FakeBuffer:
    def __init__(self, text):
        self.text = text

    def get_bounds(self):
        return 0, len(self.text)

    def get_text(self, _start, _end, _include_hidden):
        return self.text

    def set_text(self, text):
        self.text = text

    def get_end_iter(self):
        return len(self.text)

    def place_cursor(self, _position):
        pass


class _FakeEntry:
    def __init__(self, text):
        self.buffer = _FakeBuffer(text)
        self.focused = False

    def get_buffer(self):
        return self.buffer

    def grab_focus(self):
        self.focused = True


class CommandHistoryGuiTests(unittest.TestCase):
    def test_history_parser_supports_default_limit_numeric_limit_and_help(self):
        self.assertEqual(_parse_history_request("history"), {
            "help": False, "limit": memory.COMMAND_HISTORY_LIMIT,
        })
        self.assertEqual(_parse_history_request("history 25"), {
            "help": False, "limit": 25,
        })
        self.assertEqual(_parse_history_request("history 99999")["limit"], 1000)
        self.assertEqual(_parse_history_request("history help"), {
            "help": True, "limit": None,
        })
        self.assertIsNone(_parse_history_request("explain history"))

    def test_history_format_is_numbered_and_keeps_multiline_entries_on_one_row(self):
        rendered = _format_command_history([
            {"id": 12, "command_text": "first\nsecond"},
        ])

        self.assertEqual(rendered, "    12  first\\nsecond")

    @mock.patch.object(memory, "load_command_history")
    @mock.patch.object(memory, "save_command_history", return_value=True)
    def test_history_command_is_local_and_never_starts_the_agent(
        self, save_history, load_history,
    ):
        load_history.return_value = [{"id": 1, "command_text": "history"}]
        window = LiamWindow.__new__(LiamWindow)
        window.busy = False
        window.entry = _FakeEntry("history")
        window._pending_image_path = None
        window._command_history = _CommandHistoryNavigator()
        window.session_id = 9
        window._append_message = mock.Mock()
        window._clear_pending_image = mock.Mock()
        window._set_busy = mock.Mock()
        window._start_thinking = mock.Mock()

        window._on_send(None)

        save_history.assert_called_once_with("history", session_id=9)
        load_history.assert_called_once_with(memory.COMMAND_HISTORY_LIMIT)
        window._append_message.assert_has_calls([
            mock.call("history", "user"),
            mock.call("     1  history", "status"),
        ])
        window._set_busy.assert_not_called()
        window._start_thinking.assert_not_called()
        self.assertTrue(window.entry.focused)

    def test_key_handler_wires_readline_history_shortcuts(self):
        source = inspect.getsource(LiamWindow._on_entry_key_press)

        for key_name in (
            "Gdk.KEY_p", "Gdk.KEY_n", "Gdk.KEY_r", "Gdk.KEY_s",
            "Gdk.KEY_less", "Gdk.KEY_greater", "Gdk.KEY_Up", "Gdk.KEY_Down",
        ):
            self.assertIn(key_name, source)
        self.assertIn("prefix-previous", source)
        self.assertIn("prefix-next", source)
        self.assertIn("Ctrl+R", HISTORY_HELP)


if __name__ == "__main__":
    unittest.main()
