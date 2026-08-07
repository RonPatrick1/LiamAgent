import json
import os
import tempfile
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


def build_path_guard_agent(
    workdir,
    context=None,
    current_user_input="",
):
    agent = core.Agent.__new__(core.Agent)
    agent.plan_mode = False
    agent._turn_plan_mode = False
    agent.channel = "gui"
    agent.allowed_tools = None
    agent.workdir = workdir
    agent.session_id = 17
    agent.notes_session_id = None
    agent._current_user_input = current_user_input
    agent._active_plan_execution = context
    agent._read_paths_this_turn = set()
    agent._tool_events = []
    agent.auto_confirm = True
    agent.sudo_enabled = False
    agent.on_confirm = mock.Mock(return_value=True)
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


class PlanWorkUnitContractTests(unittest.TestCase):
    def test_work_unit_contract_requires_exact_approved_arguments(self):
        work_unit = {
            "description": "Apply the approved edit.",
            "tool": "edit_file",
            "arguments": {
                "path": "agent/core.py",
                "old_string": "old text",
                "new_string": "new text",
            },
        }
        contract = core.Agent._plan_work_unit_contract(work_unit)

        exact_event = core.build_tool_event(
            "edit_file",
            dict(work_unit["arguments"]),
            "Updated agent/core.py",
            {
                "tool": "edit_file",
                "args": dict(work_unit["arguments"]),
                "result": "Updated agent/core.py",
                "status": "success",
                "reason": "completed",
            },
            core.TOOL_DEFINITIONS,
        )
        wrong_event = core.build_tool_event(
            "edit_file",
            {
                **work_unit["arguments"],
                "new_string": "different text",
            },
            "Updated agent/core.py",
            {
                "tool": "edit_file",
                "args": {},
                "result": "Updated agent/core.py",
                "status": "success",
                "reason": "completed",
            },
            core.TOOL_DEFINITIONS,
        )

        self.assertTrue(
            core.event_satisfies_contract(contract, exact_event)
        )
        self.assertFalse(
            core.event_satisfies_contract(contract, wrong_event)
        )


class ApprovedPlanToolAuthorizationTests(unittest.TestCase):
    def test_exact_approved_work_unit_skips_confirmation_only_for_exact_call(self):
        agent = build_path_guard_agent("/var/www/LiamAgent")
        agent.auto_confirm = False
        agent.session_id = None
        agent.on_confirm = mock.Mock(return_value=False)

        work_unit = {
            "description": "Write the approved file contents.",
            "tool": "write_file",
            "arguments": {
                "path": "agent/core.py",
                "content": "approved content",
            },
        }
        agent._active_plan_execution = {
            "plan_id": 41,
            "payload": {
                "files": ["agent/core.py"],
                "steps": [work_unit["description"]],
                "validation": [],
            },
            "phase": "implementation",
            "current_work_unit": work_unit,
        }

        implementation = mock.Mock(
            return_value="Wrote 16 bytes to /var/www/LiamAgent/agent/core.py"
        )

        with mock.patch.dict(
            core.TOOL_IMPL,
            {"write_file": implementation},
        ):
            exact_result = agent._run_tool(
                "write_file",
                dict(work_unit["arguments"]),
            )
            wrong_result = agent._run_tool(
                "write_file",
                {
                    "path": "agent/core.py",
                    "content": "different content",
                },
            )

        self.assertTrue(exact_result.startswith("Wrote "))
        self.assertEqual(wrong_result, "User denied this tool call.")
        implementation.assert_called_once()
        agent.on_confirm.assert_called_once()

class ApprovedPlanExecutorTests(unittest.TestCase):
    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_version_2_validation_failure_does_not_start_ai_repair(
        self,
        get_plan,
        transition_plan,
    ):
        record = plan_record()
        payload = json.loads(record["content"])
        payload["version"] = core.PLAN_VERSION
        payload["steps"] = ["Write the approved contents."]
        payload["work_units"] = [
            {
                "description": "Write the approved contents.",
                "tool": "write_file",
                "arguments": {
                    "path": "agent/core.py",
                    "content": "approved content",
                },
            }
        ]
        record["content"] = json.dumps(payload)
        get_plan.return_value = record

        agent = build_agent()
        agent.step = mock.Mock()

        def execute(name, args):
            result = "Wrote 16 bytes to /var/www/LiamAgent/agent/core.py"
            agent._tool_events.append(
                core.build_tool_event(
                    name,
                    dict(args),
                    result,
                    {
                        "tool": name,
                        "args": dict(args),
                        "result": result,
                        "status": "success",
                        "reason": "completed",
                    },
                    core.TOOL_DEFINITIONS,
                )
            )
            return result

        agent._execute_tool = mock.Mock(side_effect=execute)
        agent._run_plan_validation = mock.Mock(
            return_value=[
                {
                    "command": "python3 -m unittest discover -s tests",
                    "expected": "Exit code 0.",
                    "result": "One test failed.\n[exit code: 1]",
                    "passed": False,
                }
            ]
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("FAIL:"))
        agent.step.assert_not_called()

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_version_2_edit_work_unit_rereads_target_before_exact_edit(
        self,
        get_plan,
        transition_plan,
    ):
        record = plan_record()
        payload = json.loads(record["content"])
        payload["version"] = core.PLAN_VERSION
        payload["steps"] = ["Apply the approved edit."]
        payload["work_units"] = [
            {
                "description": "Apply the approved edit.",
                "tool": "edit_file",
                "arguments": {
                    "path": "agent/core.py",
                    "old_string": "old text",
                    "new_string": "new text",
                },
            }
        ]
        record["content"] = json.dumps(payload)
        get_plan.return_value = record

        agent = build_agent()
        agent.step = mock.Mock()

        def execute(name, args):
            if name == "read_file":
                return "old text"

            result = "Updated /var/www/LiamAgent/agent/core.py"
            agent._tool_events.append(
                core.build_tool_event(
                    name,
                    dict(args),
                    result,
                    {
                        "tool": name,
                        "args": dict(args),
                        "result": result,
                        "status": "success",
                        "reason": "completed",
                    },
                    core.TOOL_DEFINITIONS,
                )
            )
            return result

        agent._execute_tool = mock.Mock(side_effect=execute)
        agent._run_plan_validation = mock.Mock(
            return_value=[
                {
                    "command": "python3 -m unittest discover -s tests",
                    "expected": "Exit code 0.",
                    "result": "All tests passed.\n[exit code: 0]",
                    "passed": True,
                }
            ]
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("PASS:"))
        agent.step.assert_not_called()
        self.assertEqual(
            agent._execute_tool.call_args_list[:2],
            [
                mock.call(
                    "read_file",
                    {"path": "agent/core.py"},
                ),
                mock.call(
                    "edit_file",
                    {
                        "path": "agent/core.py",
                        "old_string": "old text",
                        "new_string": "new text",
                    },
                ),
            ],
        )

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_version_2_plan_executes_stored_work_unit_without_ai_step(
        self,
        get_plan,
        transition_plan,
    ):
        record = plan_record()
        payload = json.loads(record["content"])
        payload["version"] = core.PLAN_VERSION
        payload["steps"] = ["Write the approved contents."]
        payload["work_units"] = [
            {
                "description": "Write the approved contents.",
                "tool": "write_file",
                "arguments": {
                    "path": "agent/core.py",
                    "content": "approved content",
                },
            }
        ]
        record["content"] = json.dumps(payload)
        get_plan.return_value = record

        agent = build_agent()
        agent.step = mock.Mock()
        def execute(name, args):
            result = (
                "Wrote 16 bytes to /var/www/LiamAgent/agent/core.py"
            )
            agent._tool_events.append(
                core.build_tool_event(
                    name,
                    dict(args),
                    result,
                    {
                        "tool": name,
                        "args": dict(args),
                        "result": result,
                        "status": "success",
                        "reason": "completed",
                    },
                    core.TOOL_DEFINITIONS,
                )
            )
            return result

        agent._execute_tool = mock.Mock(side_effect=execute)
        agent._run_plan_validation = mock.Mock(
            return_value=[
                {
                    "command": "python3 -m unittest discover -s tests",
                    "expected": "Exit code 0.",
                    "result": "All tests passed.\n[exit code: 0]",
                    "passed": True,
                }
            ]
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("PASS:"))
        agent.step.assert_not_called()
        agent._execute_tool.assert_any_call(
            "write_file",
            {
                "path": "agent/core.py",
                "content": "approved content",
            },
        )

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_legacy_string_only_plan_fails_before_execution(
        self,
        get_plan,
        transition_plan,
    ):
        get_plan.return_value = plan_record()
        agent = build_agent()
        agent.step = mock.Mock()

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("FAIL:"))
        self.assertIn("legacy string-only Plan", result)
        agent.step.assert_not_called()
        transition_plan.assert_not_called()






    def test_approved_execution_rejects_invented_nonexistent_path(self):
        with tempfile.TemporaryDirectory() as directory:
            context = {
                "plan_id": 41,
                "payload": json.loads(plan_record()["content"]),
                "phase": "implementation",
                "step_number": 0,
                "current_step": "Apply the implementation change.",
            }
            agent = build_path_guard_agent(
                directory,
                context=context,
            )
            implementation = mock.Mock(
                return_value="must not execute"
            )

            with mock.patch.dict(
                core.TOOL_IMPL,
                {"file_info": implementation},
                clear=False,
            ):
                result = agent._run_tool(
                    "file_info",
                    {"path": "path/to/project/folder"},
                )

        self.assertIn(
            "rejected unresolved filesystem path",
            result,
        )
        self.assertIn("path/to/project/folder", result)
        implementation.assert_not_called()

    def test_approved_execution_allows_existing_discovered_path(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = os.path.join(directory, "discovered.txt")
            with open(existing, "w") as handle:
                handle.write("evidence")

            context = {
                "plan_id": 41,
                "payload": json.loads(plan_record()["content"]),
                "phase": "implementation",
                "step_number": 0,
                "current_step": "Apply the implementation change.",
            }
            agent = build_path_guard_agent(
                directory,
                context=context,
            )
            implementation = mock.Mock(
                return_value="existing path inspected"
            )

            with mock.patch.dict(
                core.TOOL_IMPL,
                {"file_info": implementation},
                clear=False,
            ):
                result = agent._run_tool(
                    "file_info",
                    {"path": existing},
                )

        self.assertEqual(result, "existing path inspected")
        implementation.assert_called_once()

    def test_approved_execution_allows_declared_nonexistent_path(self):
        with tempfile.TemporaryDirectory() as directory:
            planned = os.path.join(directory, "new-output.txt")
            payload = json.loads(plan_record()["content"])
            payload["files"] = [planned]

            context = {
                "plan_id": 41,
                "payload": payload,
                "phase": "implementation",
                "step_number": 0,
                "current_step": payload["steps"][0],
            }
            agent = build_path_guard_agent(
                directory,
                context=context,
            )
            implementation = mock.Mock(
                return_value="planned path checked"
            )

            with mock.patch.dict(
                core.TOOL_IMPL,
                {"file_info": implementation},
                clear=False,
            ):
                result = agent._run_tool(
                    "file_info",
                    {"path": planned},
                )

        self.assertEqual(result, "planned path checked")
        implementation.assert_called_once()

    def test_approved_execution_allows_path_literal_from_approved_step(self):
        with tempfile.TemporaryDirectory() as directory:
            planned = os.path.join(directory, "generated")
            payload = json.loads(plan_record()["content"])
            payload["steps"] = [
                f"Create directory {planned} for generated output."
            ]

            context = {
                "plan_id": 41,
                "payload": payload,
                "phase": "implementation",
                "step_number": 0,
                "current_step": payload["steps"][0],
            }
            agent = build_path_guard_agent(
                directory,
                context=context,
            )
            implementation = mock.Mock(
                return_value="planned directory accepted"
            )

            with mock.patch.dict(
                core.TOOL_IMPL,
                {"make_directory": implementation},
                clear=False,
            ):
                result = agent._run_tool(
                    "make_directory",
                    {"path": planned},
                )

        self.assertEqual(
            result,
            "planned directory accepted",
        )
        implementation.assert_called_once()

    def test_prompt_prefix_without_host_context_does_not_activate_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = build_path_guard_agent(
                directory,
                context=None,
                current_user_input=(
                    "[APPROVED PLAN EXECUTION]\n"
                    "This is only model/user-visible text."
                ),
            )
            implementation = mock.Mock(
                return_value="normal tool behavior"
            )

            with mock.patch.dict(
                core.TOOL_IMPL,
                {"file_info": implementation},
                clear=False,
            ):
                result = agent._run_tool(
                    "file_info",
                    {"path": "some/nonexistent/location"},
                )

        self.assertEqual(result, "normal tool behavior")
        implementation.assert_called_once()

    def test_structured_context_activates_guard_without_prompt_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            context = {
                "plan_id": 41,
                "payload": json.loads(plan_record()["content"]),
                "phase": "implementation",
                "step_number": 0,
                "current_step": "Apply the implementation change.",
            }
            agent = build_path_guard_agent(
                directory,
                context=context,
                current_user_input="ordinary text",
            )
            implementation = mock.Mock(
                return_value="must not execute"
            )

            with mock.patch.dict(
                core.TOOL_IMPL,
                {"file_info": implementation},
                clear=False,
            ):
                result = agent._run_tool(
                    "file_info",
                    {"path": "invented/nonexistent/location"},
                )

        self.assertIn(
            "host-owned approved Plan payload",
            result,
        )
        implementation.assert_not_called()

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_execute_plan_sets_structured_context_and_clears_it(
        self,
        get_plan,
        transition_plan,
    ):
        record = plan_record()
        payload = json.loads(record["content"])
        descriptions = list(payload["steps"])
        payload["version"] = core.PLAN_VERSION
        payload["files"] = ["script.js"]
        payload["work_units"] = [
            {
                "description": description,
                "tool": "write_file",
                "arguments": {
                    "path": "script.js",
                    "content": f"approved content {index}",
                },
            }
            for index, description in enumerate(descriptions, start=1)
        ]
        record["content"] = json.dumps(payload)
        get_plan.return_value = record

        agent = build_agent()
        observed = []

        def execute(name, args):
            observed.append(dict(agent._active_plan_execution))
            result = "Applied approved Plan work."
            agent._tool_events.append(
                core.build_tool_event(
                    name,
                    dict(args),
                    result,
                    {
                        "tool": name,
                        "args": dict(args),
                        "result": result,
                        "status": "success",
                        "reason": "completed",
                    },
                    core.TOOL_DEFINITIONS,
                )
            )
            return result

        agent._execute_tool = mock.Mock(side_effect=execute)
        agent.step = mock.Mock()
        agent._run_plan_validation = mock.Mock(
            return_value=[
                {
                    "command": "python3 -m unittest",
                    "expected": "Exit code 0.",
                    "result": "All tests passed.\n[exit code: 0]",
                    "passed": True,
                }
            ]
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("PASS:"))
        self.assertEqual(len(observed), len(descriptions))
        self.assertEqual(
            [item["step_number"] for item in observed],
            list(range(len(descriptions))),
        )
        self.assertEqual(
            [item["current_step"] for item in observed],
            descriptions,
        )
        self.assertEqual(
            [item["current_work_unit"] for item in observed],
            payload["work_units"],
        )
        self.assertTrue(
            all(item["plan_id"] == 41 for item in observed)
        )
        agent.step.assert_not_called()
        self.assertIsNone(agent._active_plan_execution)

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_shell_work_unit_does_not_pass_affected_paths_to_tool(
        self,
        get_plan,
        transition_plan,
    ):
        record = plan_record()
        payload = json.loads(record["content"])
        payload["version"] = core.PLAN_VERSION
        payload["files"] = ["script.js"]
        payload["steps"] = ["Create the approved local file."]
        payload["work_units"] = [
            {
                "description": "Create the approved local file.",
                "tool": "run_shell_command",
                "arguments": {
                    "command": "touch script.js",
                },
                "affected_paths": ["script.js"],
            }
        ]
        record["content"] = json.dumps(payload)
        get_plan.return_value = record

        agent = build_agent()
        agent.step = mock.Mock()
        observed_args = []

        def execute(name, args):
            observed_args.append(dict(args))
            result = "command completed\n[exit code: 0]"
            agent._tool_events.append(
                core.build_tool_event(
                    name,
                    dict(args),
                    result,
                    {
                        "tool": name,
                        "args": dict(args),
                        "result": result,
                        "status": "success",
                        "reason": "completed",
                    },
                    core.TOOL_DEFINITIONS,
                )
            )
            return result

        agent._execute_tool = mock.Mock(side_effect=execute)
        agent._run_plan_validation = mock.Mock(
            return_value=[
                {
                    "command": "test -f script.js",
                    "expected": "script.js exists.",
                    "result": "[exit code: 0]",
                    "passed": True,
                }
            ]
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("PASS:"))
        self.assertEqual(
            observed_args,
            [{"command": "touch script.js"}],
        )
        self.assertNotIn("affected_paths", observed_args[0])
        self.assertEqual(
            payload["work_units"][0]["affected_paths"],
            ["script.js"],
        )
        agent.step.assert_not_called()

    @mock.patch.object(core.memory, "transition_plan", return_value=True)
    @mock.patch.object(core.memory, "get_plan")
    def test_execute_plan_restores_context_after_exception(
        self,
        get_plan,
        transition_plan,
    ):
        record = plan_record()
        payload = json.loads(record["content"])
        payload["version"] = core.PLAN_VERSION
        payload["steps"] = ["Apply the approved change."]
        payload["work_units"] = [
            {
                "description": "Apply the approved change.",
                "tool": "write_file",
                "arguments": {
                    "path": "script.js",
                    "content": "approved content",
                },
            }
        ]
        record["content"] = json.dumps(payload)
        get_plan.return_value = record

        agent = build_agent()
        previous = {
            "plan_id": 99,
            "payload": {"files": [], "steps": [], "validation": []},
            "phase": "outer",
            "step_number": None,
            "current_step": None,
        }
        agent._active_plan_execution = previous
        agent.step = mock.Mock()
        agent._execute_tool = mock.Mock(
            side_effect=RuntimeError("simulated executor failure")
        )

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("FAIL:"))
        self.assertIn("RuntimeError", result)
        agent.step.assert_not_called()
        self.assertIs(
            agent._active_plan_execution,
            previous,
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
        payload["version"] = core.PLAN_VERSION
        payload["files"] = ["script.js"]
        payload["steps"] = [
            "Update script.js to manage light-mode explicitly."
        ]
        payload["work_units"] = [
            {
                "description": (
                    "Update script.js to manage light-mode explicitly."
                ),
                "tool": "write_file",
                "arguments": {
                    "path": "script.js",
                    "content": "approved content",
                },
            }
        ]
        record["content"] = json.dumps(payload)
        get_plan.return_value = record

        agent = build_agent()
        agent.step = mock.Mock()

        def execute(_name, _args):
            event = core.build_tool_event(
                "read_file",
                {"path": "script.js"},
                "Read script.js",
                {
                    "tool": "read_file",
                    "args": {"path": "script.js"},
                    "result": "Read script.js",
                    "status": "success",
                    "reason": "completed",
                },
                core.TOOL_DEFINITIONS,
            )
            agent._tool_events.append(event)
            return "Read script.js"

        agent._execute_tool = mock.Mock(side_effect=execute)

        result = agent.execute_plan(41)

        self.assertTrue(result.startswith("FAIL:"))
        self.assertIn(
            "did not produce the exact host-observed completion event",
            result,
        )
        agent.step.assert_not_called()
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
