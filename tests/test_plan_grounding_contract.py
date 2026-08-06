import json
import tempfile
import unittest

from agent import core


def fenced(payload):
    return "```liam-plan\n" + json.dumps(payload) + "\n```"


def base_payload():
    return {
        "title": "Repair project",
        "objective": "Repair the project and verify the result.",
        "files": [],
        "steps": [
            "Apply the minimum required project repair.",
        ],
        "validation": [
            {
                "command": "python3 -m unittest discover -s tests",
                "expected": "The test suite exits successfully.",
            }
        ],
        "non_goals": [
            "Do not refactor unrelated code.",
        ],
        "risks": [
            "Existing behavior may depend on the current implementation.",
        ],
    }


def build_agent(directory, events=None):
    agent = core.Agent.__new__(core.Agent)
    agent.workdir = directory
    agent.plan_mode = True
    agent._turn_plan_mode = False
    agent._read_paths_this_turn = set()
    agent._tool_events = list(events or [])
    return agent


class PlanGroundingContractTests(unittest.TestCase):
    def test_wildcard_file_target_is_rejected(self):
        payload = base_payload()
        payload["files"] = ["/var/www/LiamApp01/*"]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn("not wildcard or glob targets", error)

    def test_unverified_service_validation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = base_payload()
            payload["validation"] = [{
                "command": "sudo systemctl status apache2",
                "expected": "Apache2 is running.",
            }]

            canonical, error = core._extract_plan_draft(
                fenced(payload)
            )
            self.assertIsNone(error)

            agent = build_agent(directory)
            problem = agent._plan_file_evidence_problem(canonical)

        self.assertIn(
            "references service 'apache2' without a same-turn "
            "verified assumption",
            problem,
        )

    def test_verified_service_assumption_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = base_payload()
            payload["validation"] = [{
                "command": "sudo systemctl status apache2",
                "expected": "Apache2 is running.",
            }]
            payload["assumptions"] = [{
                "claim": "The apache2 service is installed and active.",
                "verified_by": (
                    "run_shell_command:apache2 active"
                ),
            }]

            canonical, error = core._extract_plan_draft(
                fenced(payload)
            )
            self.assertIsNone(error)

            agent = build_agent(
                directory,
                events=[{
                    "tool": "run_shell_command",
                    "args": {
                        "command": (
                            "systemctl is-active apache2"
                        ),
                    },
                    "result": "apache2 active",
                    "status": "success",
                }],
            )
            problem = agent._plan_file_evidence_problem(canonical)

        self.assertIsNone(problem)


if __name__ == "__main__":
    unittest.main()
