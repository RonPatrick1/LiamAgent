import subprocess
import unittest
from unittest import mock

from agent import core, ssh_secrets, tools


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
        self.assertNotIn("-tt", argv)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertIsNone(run.call_args.kwargs["input"])
        self.assertIn("remote-host", result)
        self.assertTrue(result.endswith("[exit code: 0]"))

    @staticmethod
    def _identity():
        return {
            "alias": "alien",
            "hostname": "192.168.0.128",
            "port": "22",
            "user": "ronpatrick",
        }

    def test_sudo_uses_keyring_stdin_pty_and_validated_timestamp(self):
        password = "not-in-arguments"
        completed = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=f"{tools.SUDO_VALIDATED_MARKER}\r\nroot\r\n", stderr="",
        )
        with mock.patch.dict(tools.os.environ, {"LIAM_SSH_HOSTS": "alien"}), \
             mock.patch.object(tools, "_ssh_host_details", return_value=self._identity()), \
             mock.patch.object(
                 tools.ssh_secrets, "lookup_sudo_password", return_value=password,
             ) as lookup, \
             mock.patch.object(tools.subprocess, "run", return_value=completed) as run:
            result = tools.ssh_run_command(
                "alien", "id -u", timeout=15, sudo=True,
            )

        argv = run.call_args.args[0]
        remote_command = argv[-1]
        self.assertIn("-tt", argv)
        self.assertIn("sudo -S -p '' -v", remote_command)
        self.assertIn("exec </dev/null", remote_command)
        self.assertIn("sudo -n -p '' -- env", remote_command)
        self.assertIn("SYSTEMD_PAGER=cat", remote_command)
        self.assertIn("SYSTEMD_COLORS=0", remote_command)
        self.assertIn("SYSTEMD_URLIFY=0", remote_command)
        self.assertIn("PAGER=cat", remote_command)
        self.assertIn("TERM=dumb", remote_command)
        self.assertNotIn(password, " ".join(argv))
        self.assertEqual(run.call_args.kwargs["input"], f"{password}\n")
        self.assertNotIn(password, result)
        self.assertNotIn(tools.SUDO_VALIDATED_MARKER, result)
        self.assertIn("root", result)
        lookup.assert_called_once_with(
            "alien", "192.168.0.128", "22", "ronpatrick",
        )

    def test_non_boolean_sudo_value_cannot_accidentally_elevate(self):
        with mock.patch.dict(tools.os.environ, {"LIAM_SSH_HOSTS": "alien"}), \
             mock.patch.object(tools.ssh_secrets, "lookup_sudo_password") as lookup:
            with self.assertRaisesRegex(ValueError, "sudo must be true or false"):
                tools.ssh_run_command("alien", "id", sudo="false")
        lookup.assert_not_called()

    def test_missing_sudo_password_does_not_start_ssh(self):
        with mock.patch.dict(tools.os.environ, {"LIAM_SSH_HOSTS": "alien"}), \
             mock.patch.object(tools, "_ssh_host_details", return_value=self._identity()), \
             mock.patch.object(
                 tools.ssh_secrets, "lookup_sudo_password", return_value=None,
             ), \
             mock.patch.object(tools.subprocess, "run") as run:
            result = tools.ssh_run_command("alien", "id", sudo=True)

        run.assert_not_called()
        self.assertIn("no sudo password is stored", result)
        self.assertTrue(result.startswith("Error:"))

    def test_multiline_keyring_value_is_rejected_before_ssh(self):
        with mock.patch.dict(tools.os.environ, {"LIAM_SSH_HOSTS": "alien"}), \
             mock.patch.object(tools, "_ssh_host_details", return_value=self._identity()), \
             mock.patch.object(
                 tools.ssh_secrets, "lookup_sudo_password", return_value="one\ntwo",
             ), \
             mock.patch.object(tools.subprocess, "run") as run:
            result = tools.ssh_run_command("alien", "id", sudo=True)

        run.assert_not_called()
        self.assertIn("credential", result.lower())
        self.assertNotIn("one", result)

    def test_incorrect_sudo_password_is_clear_and_redacted(self):
        password = "accidental-secret"
        completed = subprocess.CompletedProcess(
            args=[], returncode=1, stdout=f"{password}\r\n",
            stderr="Sorry, try again.\r\n",
        )
        with mock.patch.dict(tools.os.environ, {"LIAM_SSH_HOSTS": "alien"}), \
             mock.patch.object(tools, "_ssh_host_details", return_value=self._identity()), \
             mock.patch.object(
                 tools.ssh_secrets, "lookup_sudo_password", return_value=password,
             ), \
             mock.patch.object(tools.subprocess, "run", return_value=completed):
            result = tools.ssh_run_command("alien", "id", sudo=True)

        self.assertIn("Sudo authentication failed", result)
        self.assertNotIn(password, result)

    def test_process_errors_are_redacted_before_returning(self):
        password = "transport-secret"
        with mock.patch.dict(tools.os.environ, {"LIAM_SSH_HOSTS": "alien"}), \
             mock.patch.object(tools, "_ssh_host_details", return_value=self._identity()), \
             mock.patch.object(
                 tools.ssh_secrets, "lookup_sudo_password", return_value=password,
             ), \
             mock.patch.object(
                 tools.subprocess, "run",
                 side_effect=OSError(f"transport rejected {password}"),
             ):
            result = tools.ssh_run_command("alien", "id", sudo=True)

        self.assertTrue(result.startswith("Error:"))
        self.assertIn("[REDACTED]", result)
        self.assertNotIn(password, result)

    def test_timeout_redacts_password_and_internal_validation_marker(self):
        password = "timeout-secret"
        expired = subprocess.TimeoutExpired(
            cmd=["ssh"], timeout=70,
            output=f"{password}\r\n{tools.SUDO_VALIDATED_MARKER}\r\n",
            stderr="WARNING: terminal is not fully functional\n",
        )
        with mock.patch.dict(tools.os.environ, {"LIAM_SSH_HOSTS": "alien"}), \
             mock.patch.object(tools, "_ssh_host_details", return_value=self._identity()), \
             mock.patch.object(
                 tools.ssh_secrets, "lookup_sudo_password", return_value=password,
             ), \
             mock.patch.object(tools.subprocess, "run", side_effect=expired):
            result = tools.ssh_run_command("alien", "systemctl show ollama", sudo=True)

        self.assertIn("timed out", result)
        self.assertIn("terminal is not fully functional", result)
        self.assertIn("[REDACTED]", result)
        self.assertNotIn(password, result)
        self.assertNotIn(tools.SUDO_VALIDATED_MARKER, result)

    def test_sudo_on_unknown_host_is_rejected_before_secret_lookup(self):
        with mock.patch.dict(tools.os.environ, {"LIAM_SSH_HOSTS": "alien"}), \
             mock.patch.object(
                 tools.ssh_secrets, "lookup_sudo_password",
             ) as lookup:
            with self.assertRaises(ValueError):
                tools.ssh_run_command("prod-rabbitmq", "id", sudo=True)
        lookup.assert_not_called()

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

    def test_sudo_is_only_a_boolean_tool_argument(self):
        schema = next(
            item for item in tools.TOOL_SCHEMAS
            if item["function"]["name"] == "ssh_run_command"
        )
        properties = schema["function"]["parameters"]["properties"]
        self.assertEqual(properties["sudo"]["type"], "boolean")
        self.assertNotIn("password", properties)

    def test_explicit_backtick_request_parses_to_secure_ssh_tool_arguments(self):
        request = (
            "On alien, run `curl -fsSL https://ollama.com/install.sh | sh` "
            "with sudo."
        )

        self.assertEqual(core._parse_explicit_ssh_command(request), {
            "host": "alien",
            "command": "curl -fsSL https://ollama.com/install.sh | sh",
            "sudo": True,
        })
        self.assertIsNone(core._parse_explicit_ssh_command(
            "On alien, install Ollama and configure it for me."
        ))

    def test_plain_sudo_request_parses_without_backticks(self):
        request = (
            "On alien, run curl -fsSL https://ollama.com/install.sh | sh "
            "with sudo."
        )

        self.assertEqual(core._parse_explicit_ssh_command(request), {
            "host": "alien",
            "command": "curl -fsSL https://ollama.com/install.sh | sh",
            "sudo": True,
        })

    def test_plain_non_sudo_request_parses_without_backticks(self):
        request = "On alien, run ollama pull llama3.1:8b."

        self.assertEqual(core._parse_explicit_ssh_command(request), {
            "host": "alien",
            "command": "ollama pull llama3.1:8b",
            "sudo": False,
        })

    def test_generic_shell_rejects_ssh_and_password_sudo_pipelines(self):
        agent = self._agent("gui")
        implementation = mock.Mock(return_value="must not run")
        commands = (
            "ssh ronpatrick@alien hostname",
            "echo 'placeholder' | sudo -S id",
        )

        with mock.patch.dict(
            core.TOOL_IMPL, {"run_shell_command": implementation}, clear=False,
        ):
            for command in commands:
                result = agent._run_tool("run_shell_command", {"command": command})
                self.assertIn("cannot invoke SSH clients or pipe a password", result)

        implementation.assert_not_called()

    def test_credential_failure_discards_unsafe_model_workaround(self):
        agent = self._agent("gui")
        trusted_error = (
            "Error: Sudo authentication failed for ronpatrick@alien. "
            "Replace the stored password in Liam's desktop settings.\n"
            "[exit code: 1]"
        )
        unsafe_model_reply = (
            "Use ssh ronpatrick@alien \"echo 'your_correct_password' | "
            "sudo -S id\""
        )

        result = agent._enforce_ssh_credential_failure(
            unsafe_model_reply, [("ssh_run_command", trusted_error)],
        )

        self.assertEqual(result, trusted_error)
        self.assertNotIn("your_correct_password", result)
        self.assertNotIn("echo", result)

    def test_noncredential_ssh_result_can_still_be_explained(self):
        agent = self._agent("gui")
        content = "The remote package was not found."

        result = agent._enforce_ssh_credential_failure(
            content,
            [("ssh_run_command", "sh: package: not found\n[exit code: 127]")],
        )

        self.assertEqual(result, content)

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    def test_explicit_desktop_request_bypasses_model_and_uses_secure_tool(
        self, _match_lessons, _save_message,
    ):
        agent = self._agent("gui")
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock(return_value="Ollama installed")
        agent._finalize_learning = mock.Mock(return_value="Ollama installed")
        agent.client.chat.side_effect = AssertionError("model must not route this request")
        request = (
            "On alien, run `curl -fsSL https://ollama.com/install.sh | sh` "
            "with sudo."
        )

        result = agent.step(request)

        expected = {
            "host": "alien",
            "command": "curl -fsSL https://ollama.com/install.sh | sh",
            "sudo": True,
        }
        self.assertEqual(result, "Ollama installed")
        agent.on_tool_call.assert_called_once_with("ssh_run_command", expected)
        agent._execute_tool.assert_called_once_with("ssh_run_command", expected)
        agent.client.chat.assert_not_called()

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    def test_plain_sudo_desktop_request_also_bypasses_model(
        self, _match_lessons, _save_message,
    ):
        agent = self._agent("gui")
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock(return_value="Ollama installed")
        agent._finalize_learning = mock.Mock(return_value="Ollama installed")
        agent.client.chat.side_effect = AssertionError("model must not route this request")
        request = (
            "On alien, run curl -fsSL https://ollama.com/install.sh | sh "
            "with sudo."
        )

        result = agent.step(request)

        expected = {
            "host": "alien",
            "command": "curl -fsSL https://ollama.com/install.sh | sh",
            "sudo": True,
        }
        self.assertEqual(result, "Ollama installed")
        agent.on_tool_call.assert_called_once_with("ssh_run_command", expected)
        agent._execute_tool.assert_called_once_with("ssh_run_command", expected)
        agent.client.chat.assert_not_called()

    @mock.patch.object(core.memory, "save_message")
    @mock.patch.object(core.memory, "match_lesson_records", return_value=[])
    def test_plain_non_sudo_desktop_request_bypasses_model(
        self, _match_lessons, _save_message,
    ):
        agent = self._agent("gui")
        agent.on_tool_call = mock.Mock()
        agent._execute_tool = mock.Mock(return_value="Model downloaded")
        agent._finalize_learning = mock.Mock(return_value="Model downloaded")
        agent.client.chat.side_effect = AssertionError("model must not route this request")
        request = "On alien, run ollama pull llama3.1:8b."

        result = agent.step(request)

        expected = {
            "host": "alien",
            "command": "ollama pull llama3.1:8b",
            "sudo": False,
        }
        self.assertEqual(result, "Model downloaded")
        agent.on_tool_call.assert_called_once_with("ssh_run_command", expected)
        agent._execute_tool.assert_called_once_with("ssh_run_command", expected)
        agent.client.chat.assert_not_called()


class SshSecretTests(unittest.TestCase):
    def test_store_uses_libsecret_without_putting_password_in_attributes(self):
        password = "keyring-only-value"
        secret_api = mock.Mock()
        secret_api.COLLECTION_DEFAULT = "default"
        secret_api.password_store_sync.return_value = True
        with mock.patch.object(ssh_secrets, "Secret", secret_api), \
             mock.patch.object(ssh_secrets, "_SCHEMA", object()):
            ssh_secrets.store_sudo_password(
                "alien", "192.168.0.128", "22", "ronpatrick", password,
            )

        store = secret_api.password_store_sync
        args = store.call_args.args
        attributes = args[1]
        label = args[3]
        self.assertNotIn(password, str(attributes))
        self.assertNotIn(password, label)
        self.assertEqual(args[4], password)

    def test_keyring_errors_do_not_include_secret(self):
        password = "never-report-me"
        secret_api = mock.Mock()
        secret_api.COLLECTION_DEFAULT = "default"
        secret_api.password_store_sync.side_effect = RuntimeError(password)
        with mock.patch.object(ssh_secrets, "Secret", secret_api), \
             mock.patch.object(ssh_secrets, "_SCHEMA", object()):
            with self.assertRaises(ssh_secrets.SudoSecretError) as caught:
                ssh_secrets.store_sudo_password(
                    "alien", "192.168.0.128", "22", "ronpatrick", password,
                )
        self.assertNotIn(password, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
