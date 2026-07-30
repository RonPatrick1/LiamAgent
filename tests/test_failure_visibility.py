import io
import json
import unittest
from unittest import mock

import LiamGUI
from agent import server


class DesktopFailureVisibilityTests(unittest.TestCase):
    def test_blank_agent_reply_is_never_sent_to_the_desktop(self):
        window = LiamGUI.LiamWindow.__new__(LiamGUI.LiamWindow)
        window.agent = mock.Mock()
        window.agent.step.return_value = ""
        window._download_reply_images = mock.Mock(return_value={})

        with mock.patch.object(LiamGUI.GLib, "idle_add") as idle_add:
            window._run_agent("do something")

        reply = idle_add.call_args.args[1]
        self.assertIn("[error]", reply)
        self.assertIn("No tool ran and no action was performed", reply)

    def test_image_preparation_failure_is_appended_to_the_text_reply(self):
        window = LiamGUI.LiamWindow.__new__(LiamGUI.LiamWindow)
        window.agent = mock.Mock()
        window.agent.step.return_value = "Here is the result."
        window._download_reply_images = mock.Mock(
            side_effect=RuntimeError("image cache unavailable"),
        )

        with mock.patch.object(LiamGUI.GLib, "idle_add") as idle_add:
            window._run_agent("do something")

        reply = idle_add.call_args.args[1]
        self.assertIn("Here is the result.", reply)
        self.assertIn("desktop failed while preparing its linked images", reply)


class MessengerFailureVisibilityTests(unittest.TestCase):
    @staticmethod
    def _handler(path, body):
        handler = server._ChatHandler.__new__(server._ChatHandler)
        handler.path = path
        encoded = json.dumps(body).encode("utf-8")
        handler.headers = {"Content-Length": str(len(encoded))}
        handler.rfile = io.BytesIO(encoded)
        handler.wfile = io.BytesIO()
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        return handler

    @mock.patch.object(server, "handle_chat", side_effect=RuntimeError("model crashed"))
    def test_chat_handler_returns_visible_reply_when_agent_raises(self, _handle_chat):
        handler = self._handler("/chat", {
            "room_id": "room", "sender_id": "sender", "message": "hello",
        })

        handler.do_POST()

        payload = json.loads(handler.wfile.getvalue())
        handler.send_response.assert_called_once_with(200)
        self.assertIn("[error]", payload["reply"])
        self.assertIn("model crashed", payload["reply"])

    @mock.patch.object(
        server, "handle_fredplayer_ask", side_effect=RuntimeError("model crashed"),
    )
    def test_fredplayer_handler_preserves_reply_shape_on_failure(self, _ask):
        handler = self._handler("/fredplayer-ask", {
            "device_id": "device", "message": "hello",
        })

        handler.do_POST()

        payload = json.loads(handler.wfile.getvalue())
        handler.send_response.assert_called_once_with(200)
        self.assertIn("[error]", payload["reply"])
        self.assertIsNone(payload["playlist"])


if __name__ == "__main__":
    unittest.main()
