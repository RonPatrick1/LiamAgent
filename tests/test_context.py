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


class HelperClientTests(unittest.TestCase):
    @staticmethod
    def _agent(helper_responses, primary_responses):
        agent = core.Agent.__new__(core.Agent)
        agent.on_status = mock.Mock()
        agent.helper_client = RecordingClient(helper_responses)
        agent.helper_text_client = agent.helper_client
        agent.client = RecordingClient(primary_responses)
        return agent

    def test_helper_success_does_not_call_primary_model(self):
        agent = self._agent(
            [{"role": "assistant", "content": '{"actionable":false}'}],
            [AssertionError("primary model should not be called")],
        )

        response = agent._helper_chat([{"role": "user", "content": "hello"}])

        self.assertEqual(response["content"], '{"actionable":false}')
        self.assertEqual(len(agent.helper_client.messages), 1)
        self.assertEqual(len(agent.client.messages), 0)

    def test_helper_transport_failure_falls_back_to_primary_model(self):
        agent = self._agent(
            [{
                "role": "assistant",
                "content": "[error] unavailable",
                "_liam_error": "transport_error",
            }],
            [{"role": "assistant", "content": '{"actionable":false}'}],
        )

        response = agent._helper_chat([{"role": "user", "content": "hello"}])

        self.assertEqual(response["content"], '{"actionable":false}')
        self.assertEqual(len(agent.client.messages), 1)
        agent.on_status.assert_called_once()

    def test_unconfigured_helper_does_not_retry_primary_error(self):
        agent = core.Agent.__new__(core.Agent)
        agent.on_status = mock.Mock()
        agent.client = RecordingClient([{
            "role": "assistant",
            "content": "[error] unavailable",
            "_liam_error": "transport_error",
        }])
        agent.helper_client = agent.client

        response = agent._helper_chat([{"role": "user", "content": "hello"}])

        self.assertEqual(response["content"], "[error] unavailable")
        self.assertEqual(len(agent.client.messages), 1)
        agent.on_status.assert_not_called()

    def test_large_result_extraction_uses_plain_text_helper(self):
        agent = core.Agent.__new__(core.Agent)
        agent.on_status = mock.Mock()
        agent.client = RecordingClient([
            AssertionError("primary model should not scan chunks"),
        ])
        agent.helper_client = RecordingClient([])
        agent.helper_text_client = RecordingClient([
            {"role": "assistant", "content": "YES"},
        ])

        result = agent._extract_from_chunk(
            "What color is the background?", "The background is black.",
        )

        self.assertEqual(result, "The background is black.")
        self.assertEqual(len(agent.helper_text_client.messages), 1)
        self.assertEqual(len(agent.client.messages), 0)


class HelperConfigurationTests(unittest.TestCase):
    @mock.patch.object(core.memory, "load_recent_messages", return_value=[])
    @mock.patch.object(core.memory, "load_recent_notes", return_value=[])
    def test_agent_builds_remote_json_helper_from_environment(
        self, _notes, _messages,
    ):
        primary = mock.Mock()
        helper = mock.Mock()
        text_helper = mock.Mock()
        environment = {
            "LIAM_HELPER_OLLAMA_URL": "http://127.0.0.1:11435/api/chat",
            "LIAM_HELPER_OLLAMA_MODEL": "llama3.1:8b",
            "LIAM_HELPER_OLLAMA_TIMEOUT": "45",
            "LIAM_HELPER_OLLAMA_KEEP_ALIVE": "30m",
        }
        with mock.patch.dict(core.os.environ, environment, clear=False), \
             mock.patch.object(
                 core, "OllamaClient", side_effect=[primary, helper, text_helper],
             ) as client_class:
            agent = core.Agent(model="liam-main")

        self.assertIs(agent.client, primary)
        self.assertIs(agent.helper_client, helper)
        self.assertIs(agent.helper_text_client, text_helper)
        self.assertEqual(client_class.call_args_list[0], mock.call(model="liam-main"))
        self.assertEqual(
            client_class.call_args_list[1],
            mock.call(
                model="llama3.1:8b",
                url="http://127.0.0.1:11435/api/chat",
                timeout=45,
                keep_alive="30m",
                options={"temperature": 0},
                response_format="json",
            ),
        )
        self.assertEqual(
            client_class.call_args_list[2],
            mock.call(
                model="llama3.1:8b",
                url="http://127.0.0.1:11435/api/chat",
                timeout=45,
                keep_alive="30m",
                options={"temperature": 0},
            ),
        )


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
    def test_helper_request_options_are_sent_without_tools(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {"role": "assistant", "content": "{}"},
        }
        client = OllamaClient(
            model="llama3.1:8b",
            url="http://alien:11434/api/chat",
            timeout=45,
            keep_alive="30m",
            options={"temperature": 0},
            response_format="json",
        )

        with mock.patch("agent.llm.requests.post", return_value=response) as post:
            result = client.chat([{"role": "user", "content": "classify"}])

        payload = post.call_args.kwargs["json"]
        self.assertEqual(result["content"], "{}")
        self.assertEqual(post.call_args.kwargs["timeout"], 45)
        self.assertEqual(payload["model"], "llama3.1:8b")
        self.assertEqual(payload["keep_alive"], "30m")
        self.assertEqual(payload["options"], {"temperature": 0})
        self.assertEqual(payload["format"], "json")
        self.assertNotIn("tools", payload)

    def test_per_call_response_format_overrides_client_default(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": '{"title":"Plan"}',
            },
        }
        schema = {
            "type": "object",
            "required": ["title"],
            "properties": {
                "title": {
                    "type": "string",
                },
            },
        }
        client = OllamaClient(response_format="json")

        with mock.patch(
            "agent.llm.requests.post",
            return_value=response,
        ) as post:
            result = client.chat(
                [{"role": "user", "content": "make a plan"}],
                response_format=schema,
            )

        payload = post.call_args.kwargs["json"]

        self.assertEqual(result["content"], '{"title":"Plan"}')
        self.assertEqual(payload["format"], schema)


    def test_transport_error_is_machine_detectable_for_fallback(self):
        with mock.patch(
            "agent.llm.requests.post",
            side_effect=requests.exceptions.ConnectTimeout("timed out"),
        ):
            result = OllamaClient(timeout=3).chat([
                {"role": "user", "content": "hello"},
            ])

        self.assertEqual(result["_liam_error"], "transport_error")
        self.assertIn("Could not reach Ollama", result["content"])

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
