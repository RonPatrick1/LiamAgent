import subprocess
import unittest
from unittest import mock

from agent import core, tools


class SshToolTests(unittest.TestCase):
    def test_configured_hosts_are_deduplicated_and_validated(self):
        with mock.patch.dict(
            tools.os.environ,
            {"LIAM_SSH_HOSTS": "worklaptop, jetson worklaptop bad/host"},
        ):
            self.assertEqual(
                tools._configured_ssh_hosts(), ["worklaptop", "jetson"]
            )
            self.assertEqual(tools._require_ssh_host("jetson"), "jetson")
            with self.assertRaises(ValueError):
                tools._require_ssh_host("prod-rabbitmq")

    def test_remote_command_uses_noninteractive_key_auth_and_no_shell(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="remote-host\n", stderr="",
        )
        with mock.patch.dict(tools.os.environ, {"LIAM_SSH_HOSTS": "jetson"}), \
             mock.patch.object(tools.subprocess, "run", return_value=completed) as run:
            result = tools.ssh_run_command("jetson", "hostname", timeout=15)

        argv = run.call_args.args[0]
        self.assertEqual(argv[-2:], ["jetson", "hostname"])
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("PasswordAuthentication=no", argv)
        self.assertIn("IdentitiesOnly=yes", argv)
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertIn("remote-host", result)
        self.assertTrue(result.endswith("[exit code: 0]"))

    def test_unknown_host_is_rejected_before_ssh_starts(self):
        with mock.patch.dict(tools.os.environ, {"LIAM_SSH_HOSTS": "jetson"}), \
             mock.patch.object(tools.subprocess, "run") as run:
            with self.assertRaises(ValueError):
                tools.ssh_run_command("prod-rabbitmq", "hostname")
        run.assert_not_called()


class DesktopOnlySshTests(unittest.TestCase):
    def _agent(self, channel):
        with mock.patch.object(core, "OllamaClient"), \
             mock.patch.object(core.memory, "load_recent_notes", return_value=[]), \
             mock.patch.object(core.memory, "load_recent_messages", return_value=[]):
            return core.Agent(channel=channel)

    def test_ssh_tools_are_advertised_only_in_gui(self):
        gui_tools = {
            schema["function"]["name"] for schema in self._agent("gui").tool_schemas
        }
        for channel in ("cli", "matrix", "fredplayer", "routine"):
            names = {
                schema["function"]["name"]
                for schema in self._agent(channel).tool_schemas
            }
            self.assertNotIn("ssh_list_hosts", names)
            self.assertNotIn("ssh_run_command", names)

        self.assertIn("ssh_list_hosts", gui_tools)
        self.assertIn("ssh_run_command", gui_tools)

    def test_non_gui_execution_is_blocked_even_without_allowed_tools(self):
        agent = self._agent("matrix")

        result = agent._run_tool("ssh_run_command", {
            "host": "jetson", "command": "hostname",
        })

        self.assertIn("only in Liam's Ubuntu desktop app", result)

    def test_remote_commands_require_confirmation_when_confirmation_is_enabled(self):
        self.assertIn("ssh_run_command", tools.DANGEROUS_TOOLS)


if __name__ == "__main__":
    unittest.main()
