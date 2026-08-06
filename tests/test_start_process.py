import unittest
from unittest import mock

from agent import core, tools
from agent.contracts import event_satisfies_contract
from agent.tools import TOOL_DEFINITIONS, TOOL_IMPL, TOOL_SCHEMAS


class StartProcessToolTests(unittest.TestCase):
    def test_start_process_is_fully_registered(self):
        self.assertIn("start_process", TOOL_IMPL)
        self.assertIn("start_process", TOOL_DEFINITIONS)
        self.assertIn("start_process", tools.DANGEROUS_TOOLS)
        self.assertIn(
            "start_process",
            {
                schema["function"]["name"]
                for schema in TOOL_SCHEMAS
            },
        )

    def test_start_process_rejects_shell_string(self):
        with mock.patch.object(tools.subprocess, "Popen") as popen:
            result = tools.start_process("ogg123 /tmp/song.flac")

        self.assertIn("argv must be a non-empty list", result)
        popen.assert_not_called()

    def test_start_process_uses_argument_array_without_shell(self):
        process = mock.Mock()
        process.pid = 4321
        process.poll.return_value = None

        with mock.patch.object(
            tools.subprocess,
            "Popen",
            return_value=process,
        ) as popen, mock.patch.object(tools.time, "sleep"):
            result = tools.start_process(
                ["ogg123", "/media/example/Ride.flac"],
                base_dir="/tmp",
            )

        self.assertEqual(
            result,
            "Started process PID 4321: ogg123 /media/example/Ride.flac",
        )
        popen.assert_called_once_with(
            ["ogg123", "/media/example/Ride.flac"],
            cwd="/tmp",
            stdin=tools.subprocess.DEVNULL,
            stdout=tools.subprocess.DEVNULL,
            stderr=tools.subprocess.DEVNULL,
            start_new_session=True,
        )

    def test_immediate_exit_is_not_reported_as_started(self):
        process = mock.Mock()
        process.pid = 4321
        process.poll.return_value = 2

        with mock.patch.object(
            tools.subprocess,
            "Popen",
            return_value=process,
        ), mock.patch.object(tools.time, "sleep"):
            result = tools.start_process(["missing-behavior"])

        self.assertIn("exited immediately with code 2", result)

    def test_classifier_emits_process_started_evidence(self):
        agent = core.Agent.__new__(core.Agent)

        event = agent._classify_tool_outcome(
            "start_process",
            {
                "argv": [
                    "ogg123",
                    "/media/example/Ride.flac",
                ],
            },
            (
                "Started process PID 4321: "
                "ogg123 /media/example/Ride.flac"
            ),
        )

        self.assertEqual(event["status"], "success")
        self.assertEqual(event["effect_kind"], "process_started")
        self.assertIn("process.start", event["capabilities"])
        self.assertEqual(
            event["targets"],
            {
                "argv": [
                    "ogg123",
                    "/media/example/Ride.flac",
                ],
            },
        )

    def test_started_process_can_complete_matching_contract(self):
        contract = {
            "required_capability": "process.start",
            "completion_mode": "process_started",
            "preferred_tool": "start_process",
            "targets": {
                "argv": [
                    "ogg123",
                    "/media/example/Ride.flac",
                ],
            },
            "constraints": {},
        }
        agent = core.Agent.__new__(core.Agent)
        event = agent._classify_tool_outcome(
            "start_process",
            {
                "argv": [
                    "ogg123",
                    "/media/example/Ride.flac",
                ],
            },
            (
                "Started process PID 4321: "
                "ogg123 /media/example/Ride.flac"
            ),
        )

        self.assertTrue(event_satisfies_contract(contract, event))

    def test_read_file_still_cannot_complete_process_start_contract(self):
        contract = {
            "required_capability": "process.start",
            "completion_mode": "process_started",
            "preferred_tool": "start_process",
            "targets": {
                "argv": [
                    "ogg123",
                    "/media/example/Ride.flac",
                ],
            },
            "constraints": {},
        }
        agent = core.Agent.__new__(core.Agent)
        event = agent._classify_tool_outcome(
            "read_file",
            {"path": "/media/example/Ride.flac"},
            "binary data",
        )

        self.assertFalse(event_satisfies_contract(contract, event))


if __name__ == "__main__":
    unittest.main()
