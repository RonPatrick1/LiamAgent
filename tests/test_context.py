import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from agent import core, tools
from agent.llm import OllamaClient


class RecordingClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages, tools=None):
        self.messages.append(messages)
        return self.responses.pop(0)


class ContextBudgetTests(unittest.TestCase):
    @staticmethod
    def _agent():
        agent = core.Agent.__new__(core.Agent)
        agent.on_status = mock.Mock()
        return agent

    def test_preflight_drops_oversized_old_tool_payload(self):
        agent = self._agent()
        raw = "RAW-TOOL-DATA" * 10_000
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "inspect the file"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "one"}]},
            {"role": "tool", "content": raw},
            {"role": "assistant", "content": "The relevant color is blue."},
            {"role": "user", "content": "Which color is the background?"},
        ]

        prepared = agent._prepare_context_messages(messages)
        combined = "\n".join(message.get("content", "") for message in prepared)

        self.assertNotIn(raw, combined)
        self.assertIn("Which color is the background?", combined)
        self.assertIn("system", combined)
        self.assertLessEqual(
            sum(agent._message_context_size(message) for message in prepared),
            core.CONTEXT_MESSAGE_CHAR_BUDGET,
        )

    def test_context_rejection_compacts_and_retries_once(self):
        agent = self._agent()
        agent.client = RecordingClient([
            {
                "role": "assistant",
                "content": "[error] context too large",
                "_liam_error": "context_overflow",
            },
            {"role": "assistant", "content": "Recovered answer"},
        ])
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "x" * 50_000},
        ]

        response = agent._chat(messages)

        self.assertEqual(response["content"], "Recovered answer")
        self.assertNotIn("_liam_error", response)
        self.assertEqual(len(agent.client.messages), 2)
        first_size = sum(
            agent._message_context_size(message) for message in agent.client.messages[0]
        )
        retry_size = sum(
            agent._message_context_size(message) for message in agent.client.messages[1]
        )
        self.assertLess(retry_size, first_size)
        self.assertLessEqual(retry_size, core.CONTEXT_RETRY_CHAR_BUDGET)

    def test_previous_tool_protocol_is_removed_before_the_next_turn(self):
        agent = self._agent()
        agent.messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "one"}]},
            {"role": "tool", "content": "large raw result"},
            {"role": "assistant", "content": "final answer"},
        ]

        agent._discard_transient_tool_history()

        self.assertEqual(
            [(message["role"], message["content"]) for message in agent.messages],
            [
                ("system", "system"),
                ("user", "read it"),
                ("assistant", "final answer"),
            ],
        )

    def test_many_relevant_chunks_still_have_a_hard_aggregate_cap(self):
        agent = self._agent()
        agent._extract_from_chunk = lambda _question, chunk: chunk

        reduced = agent._reduce_large_result(
            "find the color", "read_file", "relevant\n" * 15_000,
        )

        self.assertLessEqual(len(reduced), core.MAX_TOOL_CONTEXT_CHARS)
        self.assertIn("omitted", reduced)

    def test_raw_large_tool_result_is_reduced_before_model_history(self):
        agent = self._agent()
        raw = "raw-file-content" * 10_000
        agent._reduce_large_result = mock.Mock(return_value="bounded relevant excerpt")

        visible = agent._model_visible_tool_result(
            "find the color", "read_file", raw,
        )
        history = [{"role": "tool", "content": visible}]

        self.assertEqual(history[0]["content"], "bounded relevant excerpt")
        self.assertNotIn(raw, history[0]["content"])
        agent._reduce_large_result.assert_called_once_with(
            "find the color", "read_file", raw,
        )


class ReadFileBudgetTests(unittest.TestCase):
    def test_unrestricted_large_read_is_truncated_with_range_guidance(self):
        handle, path = tempfile.mkstemp(text=True)
        try:
            with os.fdopen(handle, "w") as file:
                file.write("x" * (tools.DEFAULT_READ_MAX_CHARS + 5000))

            result = tools.read_file(path)

            self.assertIn("read_file truncated", result)
            self.assertIn("search_text", result)
            self.assertLess(len(result), tools.DEFAULT_READ_MAX_CHARS + 500)
        finally:
            os.unlink(path)


class OllamaErrorTests(unittest.TestCase):
    def test_http_context_error_preserves_real_ollama_reason(self):
        response = mock.Mock()
        response.status_code = 400
        response.json.return_value = {
            "error": "request (33659 tokens) exceeds the available context size (32768 tokens)"
        }
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=response,
        )

        with mock.patch("agent.llm.requests.post", return_value=response):
            result = OllamaClient().chat([{"role": "user", "content": "hello"}])

        self.assertEqual(result["_liam_error"], "context_overflow")
        self.assertIn("33659 tokens", result["content"])
        self.assertNotIn("Could not reach", result["content"])


class ChatColorTests(unittest.TestCase):
    def test_chat_text_view_has_explicit_black_background(self):
        source = Path(__file__).resolve().parents[1].joinpath("LiamGUI.py").read_text()
        self.assertIn("textview.liam-chat-view text {", source)
        self.assertIn("background-color: #000000;", source)


if __name__ == "__main__":
    unittest.main()
