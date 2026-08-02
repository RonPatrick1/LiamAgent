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
        agent._tool_events = []
        return remaining.pop(0)

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
