import json
import unittest
from unittest import mock

from agent import core


def plan_record(status="approved"):
    payload = {
        "title": "Implement feature",
        "objective": "Implement and verify the requested feature.",
        "files": ["agent/core.py", "tests/test_feature.py"],
        "steps": [
            "Apply the implementation change.",
            "Add focused regression coverage.",
        ],
        "validation": [
            {
                "command": "python3 -m unittest discover -s tests",
                "expected": "Exit code 0.",
            }
        ],
        "non_goals": ["Do not refactor unrelated code."],
        "risks": ["Existing behavior may regress."],
    }
    return {
        "id": 41,
        "session_id": 17,
        "status": status,
        "content": json.dumps(payload),
    }


def build_agent():
    agent = core.Agent.__new__(core.Agent)
    agent.plan_mode = False
    agent.session_id = 17
    agent.on_status = mock.Mock()
    agent.on_tool_call = mock.Mock()
    agent._tool_events = []
    return agent


def install_step_replies(agent, replies):
    remaining = list(replies)

    def step(_prompt):
        reply = remaining.pop(0)
        agent._tool_events = [{
            "tool": "edit_file",
            "args": {},
            "result": "Applied mocked Plan step.",
            "status": "success",
            "reason": "completed",
        }]
        return reply

    agent.step = mock.Mock(side_effect=step)


def install_validation_results(agent, results):
    remaining = list(results)

    def execute(name, _args):
        result = remaining.pop(0)
        status = (
            "success"
            if result.rstrip().endswith("[exit code: 0]")
            else "failure"
        )
        agent._tool_events.append({
            "tool": name,
            "args": {},
            "result": result,
            "status": status,
            "reason": (
                "completed"
                if status == "success"
                else "nonzero_exit"
            ),
        })
        return result

    agent._execute_tool = mock.Mock(side_effect=execute)


class ApprovedPlanExecutorTests(unittest.TestCase):
    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_cycle_limit_continues_same_step_instead_of_ending_plan(
        self,
        get_plan,
        transition_plan,
    ):
        get_plan.return_value = plan_record()
        agent = build_agent()
        install_step_replies(
            agent,
            [
                "(stopped: reached the reasoning step limit "
                "without a final answer)",
                "First step complete.",
                "Second step complete.",
            ],
        )
        install_validation_results(
            agent,
            ["All tests passed.\n[exit code: 0]"],
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("PASS:"))
        self.assertEqual(agent.step.call_count, 3)
        self.assertEqual(
            transition_plan.call_args_list,
            [
                mock.call(41, "approved", "running"),
                mock.call(
                    41,
                    "running",
                    "passed",
                    result=mock.ANY,
                ),
            ],
        )

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_failed_validation_triggers_repair_and_rerun(
        self,
        get_plan,
        transition_plan,
    ):
        get_plan.return_value = plan_record()
        agent = build_agent()
        install_step_replies(
            agent,
            [
                "First step complete.",
                "Second step complete.",
                "Validation repair complete.",
            ],
        )
        install_validation_results(
            agent,
            [
                "One test failed.\n[exit code: 1]",
                "All tests passed.\n[exit code: 0]",
            ],
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("PASS:"))
        self.assertEqual(agent.step.call_count, 3)
        self.assertEqual(agent._execute_tool.call_count, 2)

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_toolless_validation_repair_retries_before_validation(
        self,
        get_plan,
        transition_plan,
    ):
        get_plan.return_value = plan_record()
        agent = build_agent()
        replies = [
            "First step complete.",
            "Second step complete.",
            "Repair described without a tool.",
            "Validation repair complete.",
        ]
        call_number = 0

        def step(_prompt):
            nonlocal call_number
            reply = replies[call_number]
            call_number += 1

            if call_number == 3:
                agent._tool_events = []
            else:
                agent._tool_events = [{
                    "tool": "edit_file",
                    "args": {},
                    "result": "Applied mocked Plan work.",
                    "status": "success",
                    "reason": "completed",
                }]

            return reply

        agent.step = mock.Mock(side_effect=step)
        install_validation_results(
            agent,
            [
                "One test failed.\n[exit code: 1]",
                "All tests passed.\n[exit code: 0]",
            ],
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("PASS:"))
        self.assertEqual(agent.step.call_count, 4)
        self.assertEqual(agent._execute_tool.call_count, 2)
        self.assertIn(
            mock.call(
                "  [approved plan validation repair produced no "
                "successful corrective action event; retrying the "
                "same repair...]"
            ),
            agent.on_status.call_args_list,
        )

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_three_toolless_validation_repairs_stop_without_rerun(
        self,
        get_plan,
        transition_plan,
    ):
        get_plan.return_value = plan_record()
        agent = build_agent()
        replies = [
            "First step complete.",
            "Second step complete.",
            "Repair prose one.",
            "Repair prose two.",
            "Repair prose three.",
        ]
        call_number = 0

        def step(_prompt):
            nonlocal call_number
            reply = replies[call_number]
            call_number += 1

            if call_number <= 2:
                agent._tool_events = [{
                    "tool": "edit_file",
                    "args": {},
                    "result": "Applied mocked Plan step.",
                    "status": "success",
                    "reason": "completed",
                }]
            else:
                agent._tool_events = []

            return reply

        agent.step = mock.Mock(side_effect=step)
        install_validation_results(
            agent,
            ["Same failure.\n[exit code: 1]"],
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("FAIL:"))
        self.assertIn(
            "validation repair produced no successful corrective "
            "action event in three consecutive attempts",
            result,
        )
        self.assertEqual(agent.step.call_count, 5)
        self.assertEqual(agent._execute_tool.call_count, 1)
        transition_plan.assert_called_with(
            41,
            "running",
            "failed",
            result=mock.ANY,
        )

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_repeated_identical_validation_failure_stops_as_failure(
        self,
        get_plan,
        transition_plan,
    ):
        get_plan.return_value = plan_record()
        agent = build_agent()
        install_step_replies(
            agent,
            [
                "First step complete.",
                "Second step complete.",
                "Repair attempt one.",
                "Repair attempt two.",
            ],
        )
        install_validation_results(
            agent,
            [
                "Same failure.\n[exit code: 1]",
                "Same failure.\n[exit code: 1]",
                "Same failure.\n[exit code: 1]",
            ],
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("FAIL:"))
        self.assertIn(
            "same failure 3 consecutive times",
            result,
        )
        transition_plan.assert_called_with(
            41,
            "running",
            "failed",
            result=mock.ANY,
        )

    def test_plan_progress_distinguishes_inspection_from_action(self):
        read_event = {
            "tool": "read_file",
            "args": {"path": "script.js"},
            "status": "success",
        }
        edit_event = {
            "tool": "edit_file",
            "args": {"path": "script.js"},
            "status": "success",
        }
        diagnostic_shell_event = {
            "tool": "run_shell_command",
            "args": {
                "command": "grep -Fq light-mode script.js"
            },
            "status": "success",
        }
        mutating_shell_event = {
            "tool": "run_shell_command",
            "args": {
                "command": "sed -i 's/dark/light/' script.js"
            },
            "status": "success",
        }
        image_event = {
            "tool": "generate_image",
            "args": {"prompt": "test"},
            "status": "success",
        }

        self.assertTrue(
            core.Agent._has_successful_plan_step_progress(
                "Inspect script.js.",
                [read_event],
            )
        )
        self.assertFalse(
            core.Agent._has_successful_plan_step_progress(
                "Update script.js.",
                [read_event],
            )
        )
        self.assertTrue(
            core.Agent._has_successful_plan_step_progress(
                "Update script.js.",
                [read_event, edit_event],
            )
        )
        self.assertFalse(
            core.Agent._has_successful_plan_step_progress(
                "Update script.js.",
                [diagnostic_shell_event],
            )
        )
        self.assertTrue(
            core.Agent._has_successful_plan_step_progress(
                "Update script.js.",
                [mutating_shell_event],
            )
        )
        self.assertTrue(
            core.Agent._has_successful_plan_step_progress(
                "Generate the requested image.",
                [image_event],
            )
        )

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_mutation_step_read_only_success_is_not_progress(
        self,
        get_plan,
        transition_plan,
    ):
        record = plan_record()
        payload = json.loads(record["content"])
        payload["files"] = ["script.js"]
        payload["steps"] = [
            "Update script.js to manage light-mode explicitly."
        ]
        record["content"] = json.dumps(payload)
        get_plan.return_value = record

        agent = build_agent()

        def failed_edit_then_read(_prompt):
            agent._tool_events = [
                {
                    "tool": "edit_file",
                    "args": {"path": "script.js"},
                    "status": "failure",
                    "reason": "edit_text_not_found",
                },
                {
                    "tool": "read_file",
                    "args": {"path": "script.js"},
                    "status": "success",
                    "reason": "completed",
                },
            ]
            return "Inspected script.js after the failed edit."

        agent.step = mock.Mock(side_effect=failed_edit_then_read)

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("FAIL:"))
        self.assertIn(
            "no qualifying Plan progress event",
            result,
        )
        self.assertEqual(agent.step.call_count, 3)
        transition_plan.assert_called_with(
            41,
            "running",
            "failed",
            result=mock.ANY,
        )

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_read_only_repair_does_not_trigger_validation_rerun(
        self,
        get_plan,
        transition_plan,
    ):
        record = plan_record()
        payload = json.loads(record["content"])
        payload["files"] = ["script.js"]
        payload["steps"] = [
            "Update script.js to manage light-mode explicitly."
        ]
        record["content"] = json.dumps(payload)
        get_plan.return_value = record

        agent = build_agent()
        call_number = 0

        def implementation_then_read_only_repairs(_prompt):
            nonlocal call_number
            call_number += 1

            if call_number == 1:
                agent._tool_events = [{
                    "tool": "edit_file",
                    "args": {"path": "script.js"},
                    "status": "success",
                    "reason": "completed",
                }]
            else:
                agent._tool_events = [{
                    "tool": "read_file",
                    "args": {"path": "script.js"},
                    "status": "success",
                    "reason": "completed",
                }]

            return "Plan execution attempt."

        agent.step = mock.Mock(
            side_effect=implementation_then_read_only_repairs
        )
        install_validation_results(
            agent,
            ["Still failing.\n[exit code: 1]"],
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("FAIL:"))
        self.assertIn(
            "validation repair produced no successful corrective "
            "action event in three consecutive attempts",
            result,
        )
        self.assertEqual(agent.step.call_count, 4)
        self.assertEqual(agent._execute_tool.call_count, 1)
        transition_plan.assert_called_with(
            41,
            "running",
            "failed",
            result=mock.ANY,
        )

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_cancellation_before_start_never_runs_agent(
        self,
        get_plan,
        transition_plan,
    ):
        get_plan.return_value = plan_record()
        agent = build_agent()
        agent.step = mock.Mock()
        cancel_event = mock.Mock()
        cancel_event.is_set.return_value = True

        result = agent.execute_plan(
            41,
            cancel_event=cancel_event,
        )

        self.assertTrue(result.startswith("SKIPPED:"))
        agent.step.assert_not_called()
        transition_plan.assert_called_once_with(
            41,
            "approved",
            "cancelled",
            result=mock.ANY,
        )

    @mock.patch.object(core.memory, "get_plan")
    def test_plan_mode_agent_cannot_execute(
        self,
        get_plan,
    ):
        agent = build_agent()
        agent.plan_mode = True

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("FAIL:"))
        self.assertIn("still restricted to Plan mode", result)
        get_plan.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
