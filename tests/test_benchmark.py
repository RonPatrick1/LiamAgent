import unittest

from scripts.benchmark_tool_routing import focused_user_text, score_response


class ToolRoutingBenchmarkTests(unittest.TestCase):
    def test_focused_shape_drops_assistant_refusal_but_keeps_user_followups(self):
        text = focused_user_text([
            {"role": "user", "content": "Check Git."},
            {"role": "assistant", "content": "I cannot access Git."},
            {"role": "user", "content": "run it yourself"},
        ])

        self.assertIn("Check Git", text)
        self.assertIn("run it yourself", text)
        self.assertNotIn("I cannot access Git", text)

    def test_score_requires_valid_expected_tool_call(self):
        case = {"expected_tools": ["read_file"]}
        response = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "LiamGUI.py"},
                },
            }],
        }

        score = score_response(case, response, {"read_file"})

        self.assertTrue(score["passed"])
        self.assertEqual(score["tool_names"], ["read_file"])

    def test_prose_instructions_do_not_count_as_tool_use(self):
        case = {"expected_tools": ["run_shell_command"]}
        response = {
            "role": "assistant",
            "content": "You should run git status yourself.",
        }

        score = score_response(case, response, {"run_shell_command"})

        self.assertFalse(score["passed"])
        self.assertTrue(score["deflected"])

    def test_excessive_tool_calls_fail_even_when_one_name_is_correct(self):
        case = {"expected_tools": ["git_status"], "max_tool_calls": 2}
        response = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "git_status", "arguments": {}}},
                {"function": {"name": "git_status", "arguments": {}}},
                {"function": {"name": "git_status", "arguments": {}}},
            ],
        }

        score = score_response(case, response, {"git_status"})

        self.assertFalse(score["passed"])
        self.assertIn("emitted 3 tool calls", score["quality_error"])


if __name__ == "__main__":
    unittest.main()
