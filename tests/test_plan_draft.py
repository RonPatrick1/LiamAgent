import json
import os
import subprocess
import unittest
from unittest import mock

from agent import core


def valid_plan():
    return {
        "title": "Add feature",
        "objective": "Implement and verify the feature.",
        "files": ["agent/core.py", "tests/test_feature.py"],
        "steps": [
            "Inspect the existing implementation.",
            "Apply the minimum code change.",
        ],
        "validation": [
            {
                "command": "python3 -m unittest discover -s tests",
                "expected": "Exit code 0.",
            }
        ],
        "non_goals": ["Do not refactor unrelated code."],
        "risks": ["Existing callers may rely on current behavior."],
    }


def fenced(payload):
    fence = chr(96) * 3
    return (
        "Plan ready.\n\n"
        + fence
        + "liam-plan\n"
        + json.dumps(payload)
        + "\n"
        + fence
    )


TEST_LISTENING_PORTS_RESULT = (
    "Current TCP listeners:\n"
    "- None found in /proc/net/tcp or /proc/net/tcp6\n"
    "Suggested currently-unused unprivileged TCP ports "
    "from 8000-8999: 8000, 8001\n"
    "These suggestions are absent from the current TCP listener "
    "table; availability must still be rechecked when the server starts."
)


class PlanDraftParserTests(unittest.TestCase):
    def test_complete_plan_is_canonicalized(self):
        canonical, error = core._extract_plan_draft(fenced(valid_plan()))
        self.assertIsNone(error)
        self.assertEqual(json.loads(canonical), valid_plan())

    def test_missing_validation_is_not_ready(self):
        payload = valid_plan()
        del payload["validation"]
        canonical, error = core._extract_plan_draft(fenced(payload))
        self.assertIsNone(canonical)
        self.assertIn("missing required fields: validation", error)

    def test_empty_validation_is_not_ready(self):
        payload = valid_plan()
        payload["validation"] = []
        canonical, error = core._extract_plan_draft(fenced(payload))
        self.assertIsNone(canonical)
        self.assertIn(
            "validation must contain at least one check",
            error,
        )

    def test_unresolved_port_placeholder_is_rejected(self):
        payload = valid_plan()
        payload["validation"][0]["command"] = (
            "curl -fsS http://192.168.0.178:<port>/"
        )

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "contains unresolved placeholder '<port>'",
            error,
        )

    def test_placeholder_validation_is_removed_when_another_remains(self):
        payload = valid_plan()
        payload["validation"].append({
            "command": "test -f TODO",
            "expected": "The additional file exists.",
        })
        content = fenced(payload)

        _canonical, error = core._extract_plan_draft(content)
        repaired = core._remove_placeholder_validation_check(
            content,
            error,
        )

        self.assertIsNotNone(repaired)
        canonical, repaired_error = core._extract_plan_draft(
            repaired
        )
        self.assertIsNone(repaired_error)
        repaired_payload = json.loads(canonical)
        self.assertEqual(
            repaired_payload["validation"],
            [payload["validation"][0]],
        )

    def test_only_placeholder_validation_is_not_removed(self):
        payload = valid_plan()
        payload["validation"] = [{
            "command": "test -f TODO",
            "expected": "The file exists.",
        }]
        content = fenced(payload)

        _canonical, error = core._extract_plan_draft(content)
        repaired = core._remove_placeholder_validation_check(
            content,
            error,
        )

        self.assertIsNone(repaired)

    def test_non_validation_placeholder_is_not_removed(self):
        payload = valid_plan()
        payload["steps"] = ["Modify TODO after inspection."]
        content = fenced(payload)

        _canonical, error = core._extract_plan_draft(content)
        repaired = core._remove_placeholder_validation_check(
            content,
            error,
        )

        self.assertIsNone(repaired)

    def test_template_path_to_project_is_rejected(self):
        payload = valid_plan()
        payload["steps"] = [
            "Inspect path/to/project/folder before implementing the fix."
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "contains unresolved placeholder 'path/to/project/folder'",
            error,
        )

    def test_file_changing_step_requires_concrete_files(self):
        payload = valid_plan()
        payload["files"] = []
        payload["steps"] = [
            "Create /var/www/LiamApp01/index.html for the local webpage."
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "files must list the concrete paths",
            error,
        )

    def test_non_goal_cannot_prohibit_plan_execution(self):
        payload = valid_plan()
        payload["non_goals"] = [
            "Creating files, starting servers, or executing the plan."
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "must not prohibit executing or implementing",
            error,
        )

    def test_fileless_non_file_plan_remains_valid(self):
        payload = valid_plan()
        payload["files"] = []
        payload["steps"] = [
            "Restart the already-configured application service."
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertEqual(
            json.loads(canonical)["files"],
            [],
        )

    def test_file_creation_non_goal_conflicts_with_steps(self):
        payload = valid_plan()
        payload["files"] = ["/var/www/LiamApp01/index.html"]
        payload["steps"] = [
            "Create /var/www/LiamApp01/index.html."
        ]
        payload["non_goals"] = [
            "Creating files."
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(canonical)
        self.assertIn(
            "conflict with file-changing implementation steps",
            error,
        )
        self.assertIn("Creating files.", error)

    def test_scoped_file_non_goal_does_not_conflict(self):
        payload = valid_plan()
        payload["files"] = [
            "/var/www/LiamApp01/index.html",
        ]
        payload["steps"] = [
            "Create /var/www/LiamApp01/index.html.",
            (
                "Run `nohup python3 -m http.server 8000 "
                "--bind 127.0.0.1 >/dev/null 2>&1 &` so it remains "
                "running during validation."
            ),
        ]
        payload["non_goals"] = [
            "Creating files outside of /var/www/LiamApp01.",
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_named_out_of_scope_file_non_goal_does_not_conflict(self):
        payload = valid_plan()
        payload["files"] = [
            "/var/www/LiamApp01/index.html",
            "/var/www/LiamApp01/style.css",
        ]
        payload["steps"] = [
            "Edit /var/www/LiamApp01/index.html.",
            "Modify /var/www/LiamApp01/style.css.",
            (
                "Run `nohup python3 -m http.server 8000 "
                "--bind 127.0.0.1 >/dev/null 2>&1 &` so it remains "
                "running during validation."
            ),
        ]
        payload["non_goals"] = [
            "Creating a separate script.js file.",
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_named_in_scope_file_non_goal_still_conflicts(self):
        payload = valid_plan()
        payload["files"] = [
            "/var/www/LiamApp01/index.html",
        ]
        payload["steps"] = [
            "Edit /var/www/LiamApp01/index.html.",
        ]
        payload["non_goals"] = [
            "Do not modify index.html.",
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(canonical)
        self.assertIn(
            "conflict with file-changing implementation steps",
            error,
        )

    def test_server_non_goal_conflicts_with_local_web_plan(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            "Start a local web server on port 8000."
        ]
        payload["non_goals"] = [
            "Starting servers."
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNotNone(canonical)
        self.assertIn(
            "conflict with the required local web server",
            error,
        )

    def test_local_web_plan_requires_serving_mechanism(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            "Create the webpage layout and styling."
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNotNone(canonical)
        self.assertIn(
            "must include a concrete server command",
            error,
        )

    def test_reuse_step_skips_serving_mechanism_requirement(self):
        payload = valid_plan()
        payload["objective"] = (
            "Restyle the local webpage already served at "
            "http://127.0.0.1:8002."
        )
        payload["steps"] = [
            "Update the page's HTML and CSS.",
            (
                "Reuse the already-running server on port 8002; "
                "no new server is needed."
            ),
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNotNone(canonical)
        self.assertIsNone(error)

    def test_generic_server_instruction_is_not_concrete(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            "Start a local web server on port 8000 in the background."
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNotNone(canonical)
        self.assertIn(
            "must include a concrete server command",
            error,
        )

    def test_foreground_server_instruction_is_rejected(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            "Run `python3 -m http.server 8000 --bind 0.0.0.0`."
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNotNone(canonical)
        self.assertIn(
            "remains running during validation",
            error,
        )

    def test_nohup_server_without_redirection_is_rejected(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            (
                "Run `nohup python3 -m http.server 8000 "
                "--bind 0.0.0.0 &`."
            ),
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNotNone(canonical)
        self.assertIn(
            "must redirect stdout and stderr",
            error,
        )

    def test_nohup_server_with_redirection_is_valid(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            (
                "Run `nohup python3 -m http.server 8000 "
                "--bind 0.0.0.0 >/dev/null 2>&1 &`."
            ),
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_assumptions_field_is_domain_neutral_not_just_web(self):
        # Deliberately a non-web example (a native C++ build) to prove the
        # assumptions field isn't special-cased to local webpage plans.
        payload = valid_plan()
        payload["objective"] = "Add a new report command to the finance CLI."
        payload["assumptions"] = [
            {
                "claim": "g++ 13 is already installed on this machine",
                "verified_by": "run_shell_command:g++ --version",
            },
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(error)
        self.assertIn("assumptions", json.loads(canonical))

    def test_assumption_missing_verified_by_is_rejected(self):
        payload = valid_plan()
        payload["assumptions"] = [
            {"claim": "The database is already reachable."},
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(canonical)
        self.assertIn("verified_by", error)

    def test_assumption_verified_by_unknown_tool_is_rejected(self):
        payload = valid_plan()
        payload["assumptions"] = [
            {
                "claim": "The database is already reachable.",
                "verified_by": "not_a_real_tool:some evidence",
            },
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(canonical)
        self.assertIn("unknown tool", error)

    def test_plan_without_assumptions_stays_unchanged(self):
        canonical, error = core._extract_plan_draft(fenced(valid_plan()))
        self.assertIsNone(error)
        self.assertNotIn("assumptions", json.loads(canonical))

    def test_local_web_plan_with_server_step_is_valid(self):
        payload = valid_plan()
        payload["objective"] = (
            "Create a local webpage at http://192.168.0.178:8000."
        )
        payload["steps"] = [
            "Create the webpage files.",
            (
                "Run `nohup python3 -m http.server 8000 "
                "--bind 0.0.0.0 >/dev/null 2>&1 &` from the webpage "
                "directory so it remains running during validation."
            ),
        ]

        canonical, error = core._extract_plan_draft(fenced(payload))

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_plain_plan_prose_is_not_stored(self):
        canonical, error = core._extract_plan_draft(
            "I still need to inspect another file."
        )
        self.assertIsNone(canonical)
        self.assertIsNone(error)

    @mock.patch.object(core.memory, "create_plan", return_value=40)
    @mock.patch.object(core.memory, "get_latest_plan", return_value=None)
    def test_visible_plan_matches_normalized_stored_plan(
        self,
        get_latest_plan,
        create_plan,
    ):
        payload = valid_plan()
        payload["validation"] = [{
            "command": (
                "grep -Fq 'transition:' style.css "
                "| grep -Fq 'smooth'"
            ),
            "expected": "Smooth transitions are present.",
        }]

        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = True
        agent._turn_plan_mode = False
        agent.session_id = 17
        agent.messages = [{"role": "assistant", "content": "original"}]

        reply = agent._capture_plan_draft(fenced(payload))

        create_plan.assert_called_once()
        stored = create_plan.call_args.args[1]
        stored_payload = json.loads(stored)

        self.assertEqual(
            stored_payload["validation"][0]["command"],
            "grep -Fq 'transition:' style.css",
        )
        self.assertIn(
            "grep -Fq 'transition:' style.css",
            reply,
        )
        self.assertNotIn(
            "| grep -Fq 'smooth'",
            reply,
        )

        visible, visible_error = core._extract_plan_draft(reply)
        self.assertIsNone(visible_error)
        self.assertEqual(visible, stored)
        self.assertEqual(agent.messages[-1]["content"], reply)
        self.assertIn(
            "[Plan draft #40 is ready for approval.]",
            reply,
        )

    @mock.patch.object(core.memory, "create_plan", return_value=41)
    @mock.patch.object(core.memory, "get_latest_plan", return_value=None)
    def test_complete_plan_is_saved_as_draft(
        self,
        get_latest_plan,
        create_plan,
    ):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = True
        agent.session_id = 17
        agent.messages = [{"role": "assistant", "content": "original"}]

        reply = agent._capture_plan_draft(fenced(valid_plan()))

        get_latest_plan.assert_called_once_with(17)
        create_plan.assert_called_once()
        self.assertEqual(create_plan.call_args.args[0], 17)
        self.assertEqual(
            json.loads(create_plan.call_args.args[1]),
            valid_plan(),
        )
        self.assertIn(
            "[Plan draft #41 is ready for approval.]",
            reply,
        )
        self.assertEqual(agent.messages[-1]["content"], reply)

    @mock.patch.object(core.memory, "create_plan")
    @mock.patch.object(core.memory, "get_latest_plan")
    def test_identical_existing_draft_is_reused(
        self,
        get_latest_plan,
        create_plan,
    ):
        canonical, _error = core._extract_plan_draft(
            fenced(valid_plan())
        )
        get_latest_plan.return_value = {
            "id": 52,
            "status": "draft",
            "content": canonical,
        }

        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = True
        agent.session_id = 17
        agent.messages = [{"role": "assistant", "content": "original"}]

        reply = agent._capture_plan_draft(fenced(valid_plan()))

        create_plan.assert_not_called()
        self.assertIn(
            "[Plan draft #52 is ready for approval.]",
            reply,
        )

    @mock.patch.object(core.memory, "create_plan")
    @mock.patch.object(core.memory, "get_latest_plan")
    def test_invalid_block_is_visible_and_not_saved(
        self,
        get_latest_plan,
        create_plan,
    ):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = True
        agent.session_id = 17
        agent.messages = [{"role": "assistant", "content": "original"}]

        fence = chr(96) * 3
        reply = agent._capture_plan_draft(
            fence + "liam-plan\nnot json\n" + fence
        )

        get_latest_plan.assert_not_called()
        create_plan.assert_not_called()
        self.assertIn("[Plan draft not saved:", reply)



    def test_validation_command_cannot_mask_failure_with_echo(self):
        payload = valid_plan()
        payload["validation"] = [
            {
                "command": (
                    "grep -qx 'PLAN TEST' plan-ui-test.txt "
                    "&& echo passed || echo failed"
                ),
                "expected": "Exit code 0 only when the content matches.",
            }
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "validation command masks failure",
            error,
        )

    def test_bare_http_probe_cannot_claim_exact_response(self):
        payload = valid_plan()
        payload["validation"] = [
            {
                "command": (
                    "curl -I http://192.168.0.178:8000"
                ),
                "expected": "HTTP/1.1 200 OK",
            }
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "must assert the exact expected protocol and status",
            error,
        )

    def test_validation_normalizes_leading_hyphen_grep_pattern(self):
        payload = valid_plan()
        payload["validation"] = [{
            "command": (
                "grep -Fq '--feature-flag:' plan-ui-test.txt"
            ),
            "expected": "The feature flag is present.",
        }]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)
        normalized = json.loads(canonical)
        self.assertEqual(
            normalized["validation"][0]["command"],
            "grep -Fq -- '--feature-flag:' plan-ui-test.txt",
        )

    def test_grep_normalization_preserves_compound_validation(self):
        payload = valid_plan()
        payload["validation"] = [{
            "command": (
                "grep -Fq '--transition-speed:' style.css "
                "&& grep -Fq 'transition:' style.css "
                "&& grep -Fq '.dark-mode {' style.css"
            ),
            "expected": (
                "Transition and theme literals remain present."
            ),
        }]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)
        normalized = json.loads(canonical)
        self.assertEqual(
            normalized["validation"][0]["command"],
            (
                "grep -Fq -- '--transition-speed:' style.css "
                "&& grep -Fq 'transition:' style.css "
                "&& grep -Fq '.dark-mode {' style.css"
            ),
        )

    def test_validation_accepts_leading_hyphen_grep_pattern_with_terminator(self):
        payload = valid_plan()
        payload["validation"] = [{
            "command": (
                "grep -Fq -- '--feature-flag:' plan-ui-test.txt"
            ),
            "expected": "The feature flag is present.",
        }]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_transition_quiet_grep_pipeline_is_normalized(self):
        payload = valid_plan()
        payload["validation"] = [{
            "command": (
                "grep -Fq 'transition:' style.css "
                "| grep -Fq 'smooth'"
            ),
            "expected": "Smooth transitions are present.",
        }]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)
        normalized = json.loads(canonical)
        self.assertEqual(
            normalized["validation"][0]["command"],
            "grep -Fq 'transition:' style.css",
        )

    def test_unrelated_quiet_grep_pipeline_is_still_rejected(self):
        payload = valid_plan()
        payload["validation"] = [{
            "command": (
                "grep -Fq 'alpha' settings.txt "
                "| grep -Fq 'beta'"
            ),
            "expected": "Both values are present.",
        }]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(canonical)
        self.assertIn(
            "pipes output from quiet grep",
            error,
        )

    def test_nonquiet_grep_pipeline_remains_valid(self):
        payload = valid_plan()
        payload["validation"] = [{
            "command": (
                "grep -F 'transition:' style.css "
                "| grep -Fq 'smooth'"
            ),
            "expected": "Smooth transitions are present.",
        }]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_redirected_background_server_without_nohup_is_rejected(self):
        payload = valid_plan()
        payload["files"] = ["index.html"]
        payload["steps"] = [
            "Update the copyright year in index.html.",
            (
                "Run `python3 -m http.server 8000 "
                "--bind 127.0.0.1 >/dev/null 2>&1 &`."
            ),
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNotNone(canonical)
        self.assertIn(
            "literal safely redirected background command",
            error,
        )

    def test_python_http_server_without_bind_is_rejected(self):
        payload = valid_plan()
        payload["files"] = ["index.html"]
        payload["steps"] = [
            "Update the copyright year in index.html.",
            (
                "Run `nohup python3 -m http.server 8000 "
                ">/dev/null 2>&1 &`."
            ),
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNotNone(canonical)
        self.assertIn(
            "must include a concrete --bind address",
            error,
        )

    def test_http_probe_with_exact_assertion_is_valid(self):
        payload = valid_plan()
        payload["validation"] = [
            {
                "command": (
                    "curl -sSI http://192.168.0.178:8000 "
                    "| grep -Fq 'HTTP/1.1 200 OK'"
                ),
                "expected": "HTTP/1.1 200 OK",
            }
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_interactive_semantic_error_uses_specialized_recovery(self):
        template = core._plan_recovery_template(
            "interactive JavaScript validation must verify addEventListener"
        )

        self.assertIs(
            template,
            core.PLAN_INTERACTIVE_JS_RECOVERY,
        )

        prompt = template.format(
            error=(
                "interactive JavaScript validation must verify "
                "'theme-toggle', addEventListener, and 'dark-mode'"
            ),
            previous_answer="invalid plan",
        )

        self.assertIn(
            "same declared JavaScript file",
            prompt,
        )
        self.assertIn(
            "Do not satisfy a requirement only by mentioning it",
            prompt,
        )
        self.assertIn("addEventListener", prompt)
        self.assertIn(
            "grep -Fq 'theme-toggle' script.js",
            prompt,
        )
        self.assertIn(
            "grep -Fq 'dark-mode' script.js",
            prompt,
        )
        self.assertIn(
            "Do not validate `theme-toggle` only against index.html",
            prompt,
        )

    def test_combined_existing_class_and_interactive_error_uses_interactive_recovery(self):
        error = (
            "implementation step claims to add CSS class(es) already present "
            "in inspected file 'style.css': 'dark-mode', 'light-mode'; "
            "reuse or modify the existing definitions instead of duplicating "
            "them; interactive JavaScript validation must verify "
            "'theme-toggle' and addEventListener"
        )

        template = core._plan_recovery_template(
            error,
            evidence_needed=False,
        )

        self.assertIs(
            template,
            core.PLAN_INTERACTIVE_JS_RECOVERY,
        )

        rendered = template.format(
            error=error,
            previous_answer="previous plan",
        )

        self.assertIn("Correct every issue", rendered)
        self.assertIn(
            "reuse or modify those definitions",
            rendered,
        )
        self.assertIn(
            "same declared JavaScript file",
            rendered,
        )

    def test_noninteractive_semantic_error_uses_generic_recovery(self):
        template = core._plan_recovery_template(
            "files contains a declared path with no implementation step"
        )

        self.assertIs(
            template,
            core.PLAN_DRAFT_RECOVERY,
        )

    def test_semantic_recovery_includes_only_inspected_file_evidence(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            inspected = os.path.join(directory, "style.css")
            uninspected = os.path.join(directory, "secret.txt")

            with open(inspected, "w") as handle:
                handle.write(
                    ".dark-mode { color: white; }\n"
                    ".light-mode { color: black; }\n"
                )
            with open(uninspected, "w") as handle:
                handle.write("must not appear\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {inspected}

            instruction = agent._with_plan_recovery_evidence(
                "repair this plan",
                canonical_plan="{}",
                evidence_needed=False,
            )

            self.assertIn(
                "Host-provided inspected repository evidence",
                instruction,
            )
            self.assertIn("style.css", instruction)
            self.assertIn(".dark-mode", instruction)
            self.assertIn(".light-mode", instruction)
            self.assertNotIn("must not appear", instruction)
            self.assertIn(
                "Do not add redundant file changes",
                instruction,
            )

    def test_structural_recovery_does_not_append_repository_evidence(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            inspected = os.path.join(directory, "style.css")
            with open(inspected, "w") as handle:
                handle.write(".dark-mode {}\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {inspected}

            instruction = agent._with_plan_recovery_evidence(
                "repair this structure",
                canonical_plan=None,
                evidence_needed=False,
            )

            self.assertEqual(
                instruction,
                "repair this structure",
            )

    def test_missing_file_evidence_recovery_does_not_append_file_contents(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            inspected = os.path.join(directory, "style.css")
            with open(inspected, "w") as handle:
                handle.write(".dark-mode {}\n")

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {inspected}

            instruction = agent._with_plan_recovery_evidence(
                "read the missing targets",
                canonical_plan="{}",
                evidence_needed=True,
            )

            self.assertEqual(
                instruction,
                "read the missing targets",
            )

    def test_likely_target_typo_uses_specialized_recovery(self):
        error = (
            "files contains nonexistent local path 'styles.css'; "
            "the inspected file 'style.css' looks like the intended target"
        )

        template = core._plan_recovery_template(
            error,
            evidence_needed=False,
        )

        self.assertIs(
            template,
            core.PLAN_TARGET_CORRECTION_RECOVERY,
        )

        rendered = template.format(
            error=error,
            previous_answer="previous plan",
        )

        self.assertIn(
            "Use the exact inspected target path",
            rendered,
        )
        self.assertIn(
            "Do not create a new similarly named file",
            rendered,
        )

    def test_target_correction_has_separate_bounded_attempt(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, "style.css")
            typo_path = os.path.join(directory, "styles.css")

            with open(style_path, "w") as handle:
                handle.write("body {}\n.dark-mode {}\n")

            invalid_shape = valid_plan()
            invalid_shape["validation"] = []

            typo_plan = valid_plan()
            typo_plan["title"] = "Update stylesheet"
            typo_plan["objective"] = "Update the existing stylesheet."
            typo_plan["files"] = [typo_path]
            typo_plan["steps"] = [
                f"Modify {typo_path}.",
            ]
            typo_plan["validation"] = [{
                "command": f"grep -Fq 'body' {typo_path}",
                "expected": "The stylesheet contains the body rule.",
            }]

            post_target_failure = valid_plan()
            post_target_failure["title"] = "Update stylesheet"
            post_target_failure["objective"] = (
                "Update the existing stylesheet."
            )
            post_target_failure["files"] = [style_path]
            post_target_failure["steps"] = [
                f"Add the dark-mode class to {style_path}.",
            ]
            post_target_failure["validation"] = [{
                "command": (
                    f"grep -Fq '.dark-mode {{' {style_path}"
                ),
                "expected": (
                    "The stylesheet contains the dark-mode selector."
                ),
            }]

            corrected_plan = valid_plan()
            corrected_plan["title"] = "Update stylesheet"
            corrected_plan["objective"] = (
                "Update the existing stylesheet."
            )
            corrected_plan["files"] = [style_path]
            corrected_plan["steps"] = [
                (
                    f"Modify {style_path} to adjust the existing "
                    "dark-mode definition for the requested behavior."
                ),
            ]
            corrected_plan["validation"] = [{
                "command": (
                    f"grep -Fq '.dark-mode {{' {style_path}"
                ),
                "expected": (
                    "The stylesheet contains the existing dark-mode selector."
                ),
            }]

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": style_path},
                        },
                    }],
                },
                {
                    "role": "assistant",
                    "content": "I will prepare the plan.",
                },
                {
                    "role": "assistant",
                    "content": fenced(invalid_shape),
                },
                {
                    "role": "assistant",
                    "content": fenced(typo_plan),
                },
                {
                    "role": "assistant",
                    "content": fenced(post_target_failure),
                },
                {
                    "role": "assistant",
                    "content": fenced(corrected_plan),
                },
            ]

            with (
                mock.patch.object(
                    core,
                    "OllamaClient",
                    return_value=client,
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_notes",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_messages",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "match_lesson_records",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "save_message",
                ),
                mock.patch.object(
                    core.memory,
                    "get_latest_plan",
                    return_value=None,
                ),
                mock.patch.object(
                    core.memory,
                    "create_plan",
                    return_value=94,
                ) as create_plan,
                mock.patch.dict(
                    os.environ,
                    {"LIAM_HELPER_OLLAMA_URL": ""},
                    clear=False,
                ),
            ):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=True,
                    workdir=directory,
                    session_id=17,
                )
                agent.on_status = mock.Mock()
                agent.on_tool_call = mock.Mock()

                reply = agent.step(
                    "Create a complete plan, but do not execute it."
                )

            self.assertEqual(client.chat.call_count, 6)
            create_plan.assert_called_once()
            self.assertIn(
                "[Plan draft #94 is ready for approval.]",
                reply,
            )

            statuses = [
                call.args[0]
                for call in agent.on_status.call_args_list
                if call.args
            ]
            self.assertTrue(
                any(
                    "retrying plan formatting (1/2)" in status
                    for status in statuses
                )
            )
            self.assertTrue(
                any(
                    "retrying plan formatting (2/2)" in status
                    for status in statuses
                )
            )
            self.assertTrue(
                any(
                    "retrying plan target (1/1)" in status
                    for status in statuses
                )
            )
            self.assertTrue(
                any(
                    "retrying post-target plan correction (1/1)" in status
                    for status in statuses
                )
            )

    def test_local_web_errors_use_specialized_recovery(self):
        error = (
            "local webpage plans must include a concrete server command; "
            "local webpage plans must explain how the server remains "
            "running during validation"
        )

        template = core._plan_recovery_template(
            error,
            evidence_needed=False,
        )

        self.assertIs(
            template,
            core.PLAN_LOCAL_WEB_RECOVERY,
        )

        rendered = template.format(
            error=error,
            previous_answer="previous plan",
        )

        self.assertIn(
            "python3 -m http.server 8000 --bind 127.0.0.1",
            rendered,
        )
        self.assertIn("Correct every issue", rendered)
        self.assertIn("missing local dependency", rendered)
        self.assertIn("steps array itself", rendered)
        self.assertIn(
            "background, detached, or service-manager mechanism",
            rendered,
        )
        self.assertIn("remains running during validation", rendered)
        self.assertIn(
            "Do not put these requirements only in validation commands",
            rendered,
        )

    def test_combined_shape_error_still_uses_local_web_recovery(self):
        error = (
            "file-changing implementation step references undeclared "
            "path(s): 'script.js'; "
            "local webpage plans must include a concrete server command"
        )

        template = core._plan_recovery_template(
            error,
            evidence_needed=False,
        )

        self.assertIs(
            template,
            core.PLAN_LOCAL_WEB_RECOVERY,
        )

    def test_interactive_recovery_requires_separate_exact_css_selectors(self):
        error = (
            "interactive JavaScript validation must verify the inspected "
            "CSS state class(es) 'dark-mode', 'light-mode'"
        )

        template = core._plan_recovery_template(
            error,
            evidence_needed=False,
        )

        self.assertIs(
            template,
            core.PLAN_INTERACTIVE_JS_RECOVERY,
        )

        rendered = template.format(
            error=error,
            previous_answer="previous plan",
        )

        self.assertIn(
            "also validate each class in the inspected CSS file",
            rendered,
        )
        self.assertIn(
            "own exact selector literal",
            rendered,
        )
        self.assertIn(
            "grep -Fq '.dark-mode {' style.css",
            rendered,
        )
        self.assertIn(
            "grep -Fq '.light-mode {' style.css",
            rendered,
        )
        self.assertIn(
            "Do not synthesize a combined selector",
            rendered,
        )
        self.assertIn(
            ".dark-mode, .light-mode",
            rendered,
        )

    def test_interactive_validation_ignores_noncontrol_section_ids(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")
            script_path = os.path.join(directory, "script.js")

            with open(index_path, "w") as handle:
                handle.write(
                    '<html>'
                    '<button id="theme-toggle">Toggle</button>'
                    '<section id="movies"></section>'
                    '<section id="tv-shows"></section>'
                    '<section id="music"></section>'
                    '<script src="script.js"></script>'
                    '</html>\n'
                )

            with open(style_path, "w") as handle:
                handle.write(
                    ".dark-mode { color: white; }\n"
                    ".light-mode { color: black; }\n"
                )

            payload = valid_plan()
            payload["objective"] = (
                "Add a theme toggle to the media library containing "
                "movies, tv-shows, and music sections."
            )
            payload["files"] = [
                index_path,
                style_path,
                script_path,
            ]
            payload["steps"] = [
                (
                    f"Modify {index_path} while retaining the movies, "
                    "tv-shows, music, and theme-toggle identifiers."
                ),
                (
                    f"Create {script_path} with addEventListener logic "
                    "connecting theme-toggle to the existing dark-mode "
                    "and light-mode classes."
                ),
                (
                    f"Verify the existing dark-mode and light-mode "
                    f"selectors in {style_path}."
                ),
            ]
            payload["validation"] = [
                {
                    "command": (
                        f"grep -Fq 'theme-toggle' {script_path} "
                        f"&& grep -Fq 'addEventListener' {script_path} "
                        f"&& grep -Fq 'dark-mode' {script_path} "
                        f"&& grep -Fq 'light-mode' {script_path}"
                    ),
                    "expected": (
                        "script.js connects the theme-toggle control "
                        "to both existing theme classes."
                    ),
                },
                {
                    "command": (
                        f"grep -Fq '.dark-mode {{' {style_path} "
                        f"&& grep -Fq '.light-mode {{' {style_path}"
                    ),
                    "expected": (
                        "Both exact existing CSS state selectors remain "
                        "present."
                    ),
                },
            ]

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent.plan_mode = False
            agent._turn_plan_mode = False
            agent._read_paths_this_turn = {
                index_path,
                style_path,
            }
            agent._tool_events = []

            problem = agent._plan_file_evidence_problem(
                json.dumps(payload)
            )

            self.assertIsNone(problem)

    def test_missing_transition_validation_is_added_from_inspected_css(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            style_path = os.path.join(directory, "style.css")
            script_path = os.path.join(directory, "script.js")

            with open(style_path, "w") as handle:
                handle.write(
                    ":root { --transition-speed: 0.3s; }\n"
                    ".dark-mode { color: white; }\n"
                    ".light-mode { color: black; }\n"
                    "body { transition: background-color "
                    "var(--transition-speed); }\n"
                )

            with open(script_path, "w") as handle:
                handle.write(
                    "themeToggle.addEventListener('click', () => {\n"
                    "  body.classList.add('dark-mode');\n"
                    "  body.classList.remove('light-mode');\n"
                    "});\n"
                )

            payload = valid_plan()
            payload["objective"] = (
                "Connect the theme toggle to dark-mode and light-mode "
                "with smooth transitions."
            )
            payload["files"] = [
                style_path,
                script_path,
            ]
            payload["steps"] = [
                (
                    f"Modify {script_path} so the theme toggle explicitly "
                    "manages dark-mode and light-mode."
                ),
                (
                    f"Preserve the existing smooth transition declarations "
                    f"in {style_path}."
                ),
            ]
            payload["validation"] = [
                {
                    "command": (
                        f"grep -Fq 'addEventListener' {script_path} && "
                        f"grep -Fq 'dark-mode' {script_path} && "
                        f"grep -Fq 'light-mode' {script_path}"
                    ),
                    "expected": (
                        "The script binds the toggle to both theme classes."
                    ),
                },
                {
                    "command": (
                        f"grep -Fq '.dark-mode {{' {style_path} && "
                        f"grep -Fq '.light-mode {{' {style_path}"
                    ),
                    "expected": (
                        "Both exact inspected CSS selectors are present."
                    ),
                },
            ]

            canonical, extraction_problem = core._extract_plan_draft(
                fenced(payload)
            )
            self.assertIsNone(extraction_problem)
            self.assertIsNotNone(canonical)

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent.plan_mode = False
            agent._turn_plan_mode = False
            agent._read_paths_this_turn = {
                style_path,
                script_path,
            }
            agent._tool_events = []

            updated_content, normalized, problem = (
                agent._normalize_plan_transition_validation(
                    fenced(payload),
                    canonical,
                )
            )

            self.assertIsNone(problem)
            normalized_payload = json.loads(normalized)
            commands = [
                check["command"]
                for check in normalized_payload["validation"]
            ]
            expected_command = (
                f"grep -Fq -- --transition-speed: {style_path} "
                f"&& grep -Fq transition: {style_path}"
            )
            self.assertIn(
                expected_command,
                commands,
            )
            self.assertIn(
                "grep -Fq -- --transition-speed:",
                updated_content,
            )
            command_result = subprocess.run(
                expected_command,
                shell=True,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                command_result.returncode,
                0,
                command_result.stderr,
            )
            self.assertEqual(
                normalized_payload["steps"],
                payload["steps"],
            )
            self.assertEqual(
                normalized_payload["non_goals"],
                payload["non_goals"],
            )

    def test_smooth_interactive_plan_requires_transition_validation(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")
            script_path = os.path.join(directory, "script.js")

            with open(index_path, "w") as handle:
                handle.write(
                    '<html><button id="theme-toggle">Toggle</button>'
                    '<script src="script.js"></script></html>\n'
                )

            with open(style_path, "w") as handle:
                handle.write(
                    ":root { --transition-speed: 0.3s; }\n"
                    ".dark-mode { color: white; }\n"
                    ".light-mode { color: black; }\n"
                    "body { transition: background-color "
                    "var(--transition-speed); }\n"
                )

            payload = valid_plan()
            payload["objective"] = (
                "Connect the theme-toggle button to dark-mode and "
                "light-mode with smooth transitions."
            )
            payload["files"] = [
                index_path,
                style_path,
                script_path,
            ]
            payload["steps"] = [
                (
                    f"Retain the theme-toggle button in {index_path}."
                ),
                (
                    f"Create {script_path} with addEventListener logic "
                    "connecting theme-toggle to the existing dark-mode "
                    "and light-mode classes."
                ),
                (
                    f"Preserve the existing smooth transition variable "
                    f"and declarations in {style_path}."
                ),
            ]
            payload["validation"] = [
                {
                    "command": (
                        f"grep -Fq 'theme-toggle' {script_path} "
                        f"&& grep -Fq 'addEventListener' {script_path} "
                        f"&& grep -Fq 'dark-mode' {script_path} "
                        f"&& grep -Fq 'light-mode' {script_path}"
                    ),
                    "expected": (
                        "The script connects the toggle control to both "
                        "theme state classes."
                    ),
                },
                {
                    "command": (
                        f"grep -Fq '.dark-mode {{' {style_path} "
                        f"&& grep -Fq '.light-mode {{' {style_path}"
                    ),
                    "expected": (
                        "Both exact theme state selectors are present."
                    ),
                },
            ]

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent.plan_mode = False
            agent._turn_plan_mode = False
            agent._read_paths_this_turn = {
                index_path,
                style_path,
            }
            agent._tool_events = []

            missing_problem = agent._plan_file_evidence_problem(
                json.dumps(payload)
            )

            self.assertIn(
                "smooth-transition CSS evidence",
                missing_problem,
            )
            self.assertIn(
                "'--transition-speed:'",
                missing_problem,
            )
            self.assertIn(
                "'transition:'",
                missing_problem,
            )

            payload["validation"].append({
                "command": (
                    f"grep -Fq -- '--transition-speed:' {style_path} "
                    f"&& grep -Fq 'transition:' {style_path}"
                ),
                "expected": (
                    "The inspected transition speed variable and a "
                    "transition declaration are present."
                ),
            })

            accepted_problem = agent._plan_file_evidence_problem(
                json.dumps(payload)
            )

            self.assertIsNone(accepted_problem)

    def test_interactive_validation_requires_connected_javascript_and_exact_css_selectors(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")
            script_path = os.path.join(directory, "script.js")

            with open(index_path, "w") as handle:
                handle.write(
                    '<html><button id="theme-toggle">Toggle</button>'
                    '<script src="script.js"></script></html>\n'
                )
            with open(style_path, "w") as handle:
                handle.write(
                    ".dark-mode { color: white; }\n"
                    ".light-mode { color: black; }\n"
                )

            payload = valid_plan()
            payload["files"] = [
                index_path,
                style_path,
                script_path,
            ]
            payload["objective"] = (
                "Connect the theme-toggle button to dark-mode and "
                "light-mode behavior."
            )
            payload["steps"] = [
                f"Modify {index_path} to retain the theme-toggle button.",
                (
                    f"Create {script_path} with addEventListener logic "
                    "that connects theme-toggle to the existing dark-mode "
                    "and light-mode classes."
                ),
                (
                    f"Verify the existing dark-mode and light-mode "
                    f"selectors in {style_path}."
                ),
            ]
            payload["validation"] = [
                {
                    "command": (
                        f"grep -Fq 'theme-toggle' {index_path}"
                    ),
                    "expected": "The HTML contains the toggle control.",
                },
                {
                    "command": (
                        f"grep -Fq 'dark-mode' {style_path} && "
                        f"grep -Fq 'light-mode' {style_path}"
                    ),
                    "expected": "The CSS contains both theme classes.",
                },
                {
                    "command": (
                        f"grep -Fq 'addEventListener' {script_path}"
                    ),
                    "expected": "The script contains an event binding.",
                },
            ]

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {
                index_path,
                style_path,
            }

            problem = agent._plan_file_evidence_problem(
                json.dumps(payload)
            )

            self.assertIsNotNone(problem)
            self.assertIn(
                "same declared JavaScript file",
                problem,
            )
            self.assertIn(
                "exact inspected CSS selector literal(s)",
                problem,
            )
            self.assertIn("'.dark-mode {'", problem)
            self.assertIn("'.light-mode {'", problem)

            payload["validation"] = [
                {
                    "command": (
                        f"grep -Fq 'theme-toggle' {index_path}"
                    ),
                    "expected": "The HTML contains the toggle control.",
                },
                {
                    "command": (
                        f"grep -Fq '.dark-mode {{' {style_path} && "
                        f"grep -Fq '.light-mode {{' {style_path}"
                    ),
                    "expected": (
                        "The CSS contains both exact state selectors."
                    ),
                },
                {
                    "command": (
                        f"grep -Fq 'theme-toggle' {script_path} && "
                        f"grep -Fq 'addEventListener' {script_path} && "
                        f"grep -Fq 'dark-mode' {script_path} && "
                        f"grep -Fq 'light-mode' {script_path}"
                    ),
                    "expected": (
                        "The same script connects the toggle control "
                        "to both state classes."
                    ),
                },
            ]

            corrected_problem = (
                agent._plan_file_evidence_problem(
                    json.dumps(payload)
                )
            )

            self.assertIsNone(corrected_problem)

    def test_unchanged_file_assertion_uses_specialized_recovery(self):
        error = (
            "validation command asserts missing content "
            "'.dark-mode, .light-mode {' in unchanged local file "
            "'/tmp/style.css'"
        )

        template = core._plan_recovery_template(
            error,
            evidence_needed=False,
        )

        self.assertIs(
            template,
            core.PLAN_UNCHANGED_FILE_ASSERTION_RECOVERY,
        )

        rendered = template.format(
            error=error,
            previous_answer="previous plan",
        )

        self.assertIn(
            "Remove the exact rejected assertion",
            rendered,
        )
        self.assertIn(
            "do not repeat it unchanged",
            rendered,
        )
        self.assertIn(
            "inspected-file evidence",
            rendered,
        )
        self.assertIn(
            "Do not add an unchanged file",
            rendered,
        )

    def test_other_semantic_errors_still_use_generic_recovery(self):
        template = core._plan_recovery_template(
            "validation must contain at least one check",
            evidence_needed=False,
        )

        self.assertIs(
            template,
            core.PLAN_DRAFT_RECOVERY,
        )

    def test_semantic_recovery_prompt_requires_content_correction(self):
        prompt = core.PLAN_DRAFT_RECOVERY.format(
            error=(
                "interactive JavaScript validation must verify "
                "'theme-toggle', addEventListener, and 'dark-mode'"
            ),
            previous_answer="invalid plan",
        )

        self.assertIn(
            "Semantic errors require changing the plan steps or "
            "validation commands",
            prompt,
        )
        self.assertIn("'theme-toggle'", prompt)
        self.assertIn("addEventListener", prompt)
        self.assertIn("'dark-mode'", prompt)
        self.assertIn(
            "Do not return the same invalid content unchanged",
            prompt,
        )

    def test_invalid_plan_response_is_corrected_by_one_model_retry(self):
        invalid = valid_plan()
        invalid.pop("non_goals")
        invalid.pop("risks")
        invalid["validation"] = (
            "grep -qx 'PLAN TEST' plan-ui-test.txt "
            "&& echo passed || echo failed"
        )

        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": fenced(invalid),
            },
            {
                "role": "assistant",
                "content": json.dumps(valid_plan()),
            },
        ]

        with (
            mock.patch.object(
                core,
                "OllamaClient",
                return_value=client,
            ),
            mock.patch.object(
                core.memory,
                "load_recent_notes",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "load_recent_messages",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "match_lesson_records",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "save_message",
            ),
            mock.patch.object(
                core.memory,
                "get_latest_plan",
                return_value=None,
            ),
            mock.patch.object(
                core.memory,
                "create_plan",
                return_value=41,
            ) as create_plan,
            mock.patch.dict(
                os.environ,
                {"LIAM_HELPER_OLLAMA_URL": ""},
                clear=False,
            ),
        ):
            agent = core.Agent(
                channel="gui",
                plan_mode=True,
                workdir="/var/www/LiamApp01",
                session_id=17,
            )
            agent.on_status = mock.Mock()
            reply = agent.step(
                "Create a complete plan, but do not execute it."
            )

        self.assertEqual(client.chat.call_count, 2)
        self.assertEqual(
            client.chat.call_args_list[1].kwargs["response_format"],
            core.PLAN_DRAFT_JSON_SCHEMA,
        )
        agent.on_status.assert_called_once()
        self.assertIn(
            "retrying plan formatting (1/2)",
            agent.on_status.call_args.args[0],
        )
        self.assertNotIn(
            "tool selection",
            agent.on_status.call_args.args[0],
        )
        create_plan.assert_called_once()
        self.assertIn(
            "[Plan draft #41 is ready for approval.]",
            reply,
        )
        self.assertNotIn(
            "[Plan draft not saved:",
            reply,
        )

    @mock.patch.object(core.memory, "create_plan")
    @mock.patch.object(core.memory, "get_latest_plan")
    def test_capture_revalidates_file_evidence_before_storage(
        self,
        get_latest_plan,
        create_plan,
    ):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "plan-ui-test.txt")
            with open(target, "w") as handle:
                handle.write("PLAN TEST\n")

            agent = core.Agent.__new__(core.Agent)
            agent.plan_mode = True
            agent.session_id = 17
            agent.workdir = directory
            agent._read_paths_this_turn = set()
            agent.messages = [
                {"role": "assistant", "content": "original"},
            ]

            payload = valid_plan()
            payload["files"] = [target]
            payload["steps"] = [
                "Modify plan-ui-test.txt.",
            ]

            reply = agent._capture_plan_draft(
                fenced(payload)
            )

        get_latest_plan.assert_not_called()
        create_plan.assert_not_called()
        self.assertIn("[Plan draft not saved:", reply)
        self.assertIn("not inspected with read_file", reply)

    def test_extract_error_is_not_erased_by_file_evidence(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            canonical = json.dumps(
                valid_plan(),
                sort_keys=True,
            )
            semantic_error = (
                "local webpage plans must include a concrete server command"
            )

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": fenced(valid_plan()),
                },
                {
                    "role": "assistant",
                    "content": fenced(valid_plan()),
                },
                {
                    "role": "assistant",
                    "content": fenced(valid_plan()),
                },
            ]

            with (
                mock.patch.object(
                    core,
                    "OllamaClient",
                    return_value=client,
                ),
                mock.patch.object(
                    core,
                    "_extract_plan_draft",
                    return_value=(canonical, semantic_error),
                ),
                mock.patch.object(
                    core.Agent,
                    "_plan_file_evidence_problem",
                    return_value=None,
                ) as file_evidence,
                mock.patch.object(
                    core.memory,
                    "load_recent_notes",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_messages",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "match_lesson_records",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "save_message",
                ),
                mock.patch.object(
                    core.memory,
                    "get_latest_plan",
                    return_value=None,
                ),
                mock.patch.object(
                    core.memory,
                    "create_plan",
                ) as create_plan,
                mock.patch.dict(
                    os.environ,
                    {"LIAM_HELPER_OLLAMA_URL": ""},
                    clear=False,
                ),
            ):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=True,
                    workdir=directory,
                    session_id=17,
                )
                agent.on_status = mock.Mock()

                reply = agent.step(
                    "Create a complete plan, but do not execute it."
                )

            self.assertEqual(client.chat.call_count, 3)
            self.assertEqual(file_evidence.call_count, 3)
            create_plan.assert_not_called()
            self.assertIn(
                "[Plan draft not saved after 2 formatting retries:",
                reply,
            )
            self.assertIn(semantic_error, reply)

            statuses = [
                call.args[0]
                for call in agent.on_status.call_args_list
                if call.args
            ]
            self.assertTrue(
                any(
                    "retrying plan formatting (1/2)" in status
                    for status in statuses
                )
            )
            self.assertTrue(
                any(
                    "retrying plan formatting (2/2)" in status
                    for status in statuses
                )
            )

    def test_plan_rejects_undeclared_file_in_change_step(self):
        payload = valid_plan()
        payload["files"] = ["settings.txt"]
        payload["steps"] = [
            "Modify settings.txt.",
            "Create helper.py.",
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNotNone(canonical)
        self.assertIn(
            "file-changing implementation step references "
            "undeclared path(s)",
            error,
        )
        self.assertIn("'helper.py'", error)
        self.assertIn(
            "every created or modified file must be listed in files",
            error,
        )

    def test_single_segment_slash_value_is_not_a_file_path(self):
        payload = valid_plan()
        payload["files"] = ["settings.txt"]
        payload["steps"] = [
            "Modify settings.txt to store /dark as the selected theme value.",
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_two_component_absolute_path_remains_a_file_path(self):
        payload = valid_plan()
        payload["files"] = ["/etc/hosts"]
        payload["steps"] = [
            "Modify /etc/hosts.",
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_declared_absolute_path_matches_sentence_ending_period(self):
        payload = valid_plan()
        payload["files"] = ["/tmp/example/index.html"]
        payload["steps"] = [
            "Create /tmp/example/index.html.",
            (
                "Run `nohup python3 -m http.server 8000 "
                "--bind 127.0.0.1 >/dev/null 2>&1 &` so it remains "
                "running during validation."
            ),
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_local_web_rejects_background_prose_without_persistent_command(self):
        payload = valid_plan()
        payload["files"] = ["index.html"]
        payload["steps"] = [
            "Update the copyright year in index.html.",
            (
                "Run `python3 -m http.server 8000 "
                "--bind 127.0.0.1` in the background during validation."
            ),
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNotNone(canonical)
        self.assertIn(
            "using a literal safely redirected background command",
            error,
        )

    def test_html_target_requires_local_server_without_web_wording(self):
        payload = valid_plan()
        payload["files"] = ["index.html"]
        payload["steps"] = [
            "Update the copyright year in index.html.",
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNotNone(canonical)
        self.assertIn(
            "local webpage plans must include a concrete server command",
            error,
        )
        self.assertIn(
            "local webpage plans must explain how the server remains "
            "running during validation",
            error,
        )

    def test_related_plan_shape_errors_are_reported_together(self):
        payload = valid_plan()
        payload["files"] = [
            "index.html",
            "style.css",
        ]
        payload["steps"] = [
            "Update the copyright year in index.html.",
            "Add or modify JavaScript code in script.js.",
            "Verify existing theme styles in style.css.",
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNotNone(canonical)
        self.assertIn("'script.js'", error)
        self.assertIn(
            "local webpage plans must include a concrete server command",
            error,
        )
        self.assertIn(
            "local webpage plans must explain how the server remains "
            "running during validation",
            error,
        )

    def test_html_target_accepts_concrete_background_server_step(self):
        payload = valid_plan()
        payload["files"] = ["index.html"]
        payload["steps"] = [
            "Update the copyright year in index.html.",
            (
                "Start `nohup python3 -m http.server 8000 "
                "--bind 127.0.0.1 >/dev/null 2>&1 &` so it remains "
                "running during validation."
            ),
        ]

        canonical, error = core._extract_plan_draft(
            fenced(payload)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(canonical)

    def test_evidence_recovery_does_not_consume_formatting_budget(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")

            with open(index_path, "w") as handle:
                handle.write("<html></html>\n")
            with open(style_path, "w") as handle:
                handle.write("body {}\n")

            server_step = (
                "Run `nohup python3 -m http.server 8000 "
                "--bind 127.0.0.1 >/dev/null 2>&1 &` so it remains "
                "running during validation."
            )

            inspected_plan = valid_plan()
            inspected_plan["files"] = [
                index_path,
                style_path,
            ]
            inspected_plan["steps"] = [
                f"Modify {index_path}.",
                f"Modify {style_path}.",
                server_step,
            ]

            semantic_failure = valid_plan()
            semantic_failure["files"] = [
                index_path,
                style_path,
            ]
            semantic_failure["steps"] = [
                f"Modify {index_path}.",
                server_step,
            ]

            corrected_plan = valid_plan()
            corrected_plan["files"] = [
                index_path,
                style_path,
            ]
            corrected_plan["steps"] = [
                f"Modify {index_path}.",
                f"Modify {style_path}.",
                server_step,
            ]

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": "I need to inspect the project first.",
                },
                {
                    "role": "assistant",
                    "content": fenced(inspected_plan),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": index_path},
                            },
                        },
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": style_path},
                            },
                        },
                        {
                            "function": {
                                "name": "listening_ports",
                                "arguments": {},
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": fenced(semantic_failure),
                },
                {
                    "role": "assistant",
                    "content": fenced(corrected_plan),
                },
            ]

            with (
                mock.patch.object(
                    core,
                    "OllamaClient",
                    return_value=client,
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_notes",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_messages",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "match_lesson_records",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "save_message",
                ),
                mock.patch.object(
                    core.memory,
                    "get_latest_plan",
                    return_value=None,
                ),
                mock.patch.object(
                    core.memory,
                    "create_plan",
                    return_value=93,
                ) as create_plan,
                mock.patch.dict(
                    core.TOOL_IMPL,
                    {
                        "listening_ports": (
                            lambda **_kwargs:
                            TEST_LISTENING_PORTS_RESULT
                        ),
                    },
                    clear=False,
                ),
                mock.patch.dict(
                    os.environ,
                    {"LIAM_HELPER_OLLAMA_URL": ""},
                    clear=False,
                ),
            ):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=True,
                    workdir=directory,
                    session_id=17,
                )
                agent.on_status = mock.Mock()
                agent.on_tool_call = mock.Mock()

                reply = agent.step(
                    "Create a complete plan, but do not execute it."
                )

            self.assertEqual(client.chat.call_count, 5)
            create_plan.assert_called_once()
            self.assertIn(
                "[Plan draft #93 is ready for approval.]",
                reply,
            )

            statuses = [
                call.args[0]
                for call in agent.on_status.call_args_list
                if call.args
            ]
            self.assertTrue(
                any(
                    "retrying plan formatting (1/2)" in status
                    for status in statuses
                )
            )
            self.assertTrue(
                any(
                    "retrying plan evidence (1/1)" in status
                    for status in statuses
                )
            )
            self.assertTrue(
                any(
                    "retrying plan formatting (2/2)" in status
                    for status in statuses
                )
            )

    def test_exhausted_formatting_budget_allows_one_post_evidence_semantic_correction(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            style_path = os.path.join(directory, "style.css")

            with open(index_path, "w") as handle:
                handle.write("<html></html>\n")
            with open(style_path, "w") as handle:
                handle.write("body {}\n")

            server_step = (
                "Run `nohup python3 -m http.server 8000 "
                "--bind 127.0.0.1 >/dev/null 2>&1 &` so it remains "
                "running during validation."
            )

            local_web_failure = valid_plan()
            local_web_failure["files"] = [index_path]
            local_web_failure["steps"] = [
                f"Modify {index_path}.",
            ]

            evidence_plan = valid_plan()
            evidence_plan["files"] = [
                index_path,
                style_path,
            ]
            evidence_plan["steps"] = [
                f"Modify {index_path}.",
                f"Modify {style_path}.",
                server_step,
            ]

            post_evidence_failure = valid_plan()
            post_evidence_failure["files"] = [
                index_path,
                style_path,
            ]
            post_evidence_failure["steps"] = [
                f"Modify {index_path}.",
                f"Modify {style_path}.",
            ]

            corrected_plan = valid_plan()
            corrected_plan["files"] = [
                index_path,
                style_path,
            ]
            corrected_plan["steps"] = [
                f"Modify {index_path}.",
                f"Modify {style_path}.",
                server_step,
            ]

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": index_path},
                        },
                    }],
                },
                {
                    "role": "assistant",
                    "content": "I will prepare the plan.",
                },
                {
                    "role": "assistant",
                    "content": fenced(local_web_failure),
                },
                {
                    "role": "assistant",
                    "content": fenced(evidence_plan),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": style_path},
                            },
                        },
                        {
                            "function": {
                                "name": "listening_ports",
                                "arguments": {},
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": fenced(post_evidence_failure),
                },
                {
                    "role": "assistant",
                    "content": fenced(corrected_plan),
                },
            ]

            with (
                mock.patch.object(
                    core,
                    "OllamaClient",
                    return_value=client,
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_notes",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_messages",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "match_lesson_records",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "save_message",
                ),
                mock.patch.object(
                    core.memory,
                    "get_latest_plan",
                    return_value=None,
                ),
                mock.patch.object(
                    core.memory,
                    "create_plan",
                    return_value=95,
                ) as create_plan,
                mock.patch.dict(
                    core.TOOL_IMPL,
                    {
                        "listening_ports": (
                            lambda **_kwargs:
                            TEST_LISTENING_PORTS_RESULT
                        ),
                    },
                    clear=False,
                ),
                mock.patch.dict(
                    os.environ,
                    {"LIAM_HELPER_OLLAMA_URL": ""},
                    clear=False,
                ),
            ):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=True,
                    workdir=directory,
                    session_id=17,
                )
                agent.on_status = mock.Mock()
                agent.on_tool_call = mock.Mock()

                reply = agent.step(
                    "Create a complete plan, but do not execute it."
                )

            self.assertEqual(client.chat.call_count, 7)
            create_plan.assert_called_once()
            self.assertIn(
                "[Plan draft #95 is ready for approval.]",
                reply,
            )

            statuses = [
                call.args[0]
                for call in agent.on_status.call_args_list
                if call.args
            ]
            self.assertTrue(
                any(
                    "retrying plan formatting (1/2)" in status
                    for status in statuses
                )
            )
            self.assertTrue(
                any(
                    "retrying plan formatting (2/2)" in status
                    for status in statuses
                )
            )
            self.assertTrue(
                any(
                    "retrying plan evidence (1/1)" in status
                    for status in statuses
                )
            )
            self.assertTrue(
                any(
                    "retrying post-evidence plan correction (1/1)"
                    in status
                    for status in statuses
                )
            )

    def test_missing_html_dependency_step_accepts_sentence_ending_period(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            script_path = os.path.join(directory, "script.js")

            with open(index_path, "w") as handle:
                handle.write(
                    '<html><script src="script.js"></script></html>\n'
                )

            payload = valid_plan()
            payload["files"] = [
                index_path,
                script_path,
            ]
            payload["steps"] = [
                f"Modify {index_path}.",
                f"Create {script_path}.",
            ]

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = {index_path}

            problem = agent._plan_file_evidence_problem(payload)

            self.assertIsNone(problem)

    def test_shape_and_repository_semantic_errors_are_reported_together(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")
            script_path = os.path.join(directory, "script.js")

            with open(index_path, "w") as handle:
                handle.write(
                    '<html><script src="script.js"></script></html>\n'
                )

            invalid_plan = valid_plan()
            invalid_plan["title"] = "Update page"
            invalid_plan["objective"] = "Update the local webpage."
            invalid_plan["files"] = [index_path]
            invalid_plan["steps"] = [
                f"Modify {index_path}.",
            ]
            invalid_plan["validation"] = [{
                "command": f"grep -Fq 'script.js' {index_path}",
                "expected": "The page references script.js.",
            }]

            corrected_plan = valid_plan()
            corrected_plan["title"] = "Update page"
            corrected_plan["objective"] = "Update the local webpage."
            corrected_plan["files"] = [
                index_path,
                script_path,
            ]
            corrected_plan["steps"] = [
                f"Modify {index_path}.",
                f"Create {script_path}.",
                (
                    "Run `nohup python3 -m http.server 8000 "
                    "--bind 127.0.0.1 >/dev/null 2>&1 &` so it remains "
                    "running during validation."
                ),
            ]
            corrected_plan["validation"] = [
                {
                    "command": f"grep -Fq 'script.js' {index_path}",
                    "expected": "The page references script.js.",
                },
                {
                    "command": f"test -f {script_path}",
                    "expected": "script.js exists.",
                },
            ]

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": index_path},
                            },
                        },
                        {
                            "function": {
                                "name": "listening_ports",
                                "arguments": {},
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": fenced(invalid_plan),
                },
                {
                    "role": "assistant",
                    "content": fenced(corrected_plan),
                },
            ]

            with (
                mock.patch.object(
                    core,
                    "OllamaClient",
                    return_value=client,
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_notes",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_messages",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "match_lesson_records",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "save_message",
                ),
                mock.patch.object(
                    core.memory,
                    "get_latest_plan",
                    return_value=None,
                ),
                mock.patch.object(
                    core.memory,
                    "create_plan",
                    return_value=96,
                ) as create_plan,
                mock.patch.dict(
                    core.TOOL_IMPL,
                    {
                        "listening_ports": (
                            lambda **_kwargs:
                            TEST_LISTENING_PORTS_RESULT
                        ),
                    },
                    clear=False,
                ),
                mock.patch.dict(
                    os.environ,
                    {"LIAM_HELPER_OLLAMA_URL": ""},
                    clear=False,
                ),
            ):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=True,
                    workdir=directory,
                    session_id=17,
                )
                agent.on_status = mock.Mock()
                agent.on_tool_call = mock.Mock()

                reply = agent.step(
                    "Create a complete plan, but do not execute it."
                )

            self.assertEqual(client.chat.call_count, 3)
            create_plan.assert_called_once()
            self.assertIn(
                "[Plan draft #96 is ready for approval.]",
                reply,
            )

            statuses = [
                call.args[0]
                for call in agent.on_status.call_args_list
                if call.args
            ]
            combined_statuses = [
                status
                for status in statuses
                if "local webpage plans must include" in status
            ]
            self.assertEqual(len(combined_statuses), 1)
            self.assertIn(
                "references missing local dependency 'script.js'",
                combined_statuses[0],
            )
            self.assertIn(
                "retrying plan formatting (1/2)",
                combined_statuses[0],
            )

    def test_local_web_plan_requires_matching_listening_ports_evidence(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")

            payload = valid_plan()
            payload["objective"] = "Create a local webpage."
            payload["files"] = [index_path]
            payload["steps"] = [
                f"Create {index_path}.",
                (
                    "Run `nohup python3 -m http.server 8000 "
                    "--bind 127.0.0.1 >/dev/null 2>&1 &` so it remains "
                    "running during validation."
                ),
            ]
            payload["validation"] = [{
                "command": f"test -f {index_path}",
                "expected": "index.html exists.",
            }]

            canonical, error = core._extract_plan_draft(
                fenced(payload)
            )

            self.assertIsNone(error)
            self.assertIsNotNone(canonical)

            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent.plan_mode = True
            agent._turn_plan_mode = False
            agent._read_paths_this_turn = set()
            agent._tool_events = []

            missing_problem = (
                agent._plan_file_evidence_problem(canonical)
            )

            self.assertIn(
                "requires listening_ports evidence this turn",
                missing_problem,
            )
            self.assertIn("8000", missing_problem)

            agent._tool_events = [{
                "tool": "listening_ports",
                "status": "success",
                "result": (
                    "Current TCP listeners:\n"
                    "- None found in /proc/net/tcp or /proc/net/tcp6\n"
                    "Suggested currently-unused unprivileged TCP ports "
                    "from 8000-8999: 8000, 8001\n"
                    "These suggestions are absent from the current TCP "
                    "listener table; availability must still be "
                    "rechecked when the server starts."
                ),
            }]

            accepted_problem = (
                agent._plan_file_evidence_problem(canonical)
            )

            self.assertIsNone(accepted_problem)

            agent._tool_events[0]["result"] = (
                "Current TCP listeners:\n"
                "- tcp 127.0.0.1:8000\n"
                "Suggested currently-unused unprivileged TCP ports "
                "from 8000-8999: 8001, 8002\n"
                "These suggestions are absent from the current TCP "
                "listener table; availability must still be rechecked "
                "when the server starts."
            )

            mismatch_problem = (
                agent._plan_file_evidence_problem(canonical)
            )

            self.assertIn(
                "selected port(s): 8000",
                mismatch_problem,
            )
            self.assertIn(
                "suggested currently-unused port(s): 8001, 8002",
                mismatch_problem,
            )

    def test_plan_recovery_includes_latest_successful_port_evidence(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            agent = core.Agent.__new__(core.Agent)
            agent.workdir = directory
            agent._read_paths_this_turn = set()
            agent._tool_events = [
                {
                    "tool": "listening_ports",
                    "status": "success",
                    "result": (
                        "Suggested currently-unused unprivileged TCP ports "
                        "from 8000-8999: 8001, 8002"
                    ),
                },
                {
                    "tool": "read_file",
                    "status": "success",
                    "result": "unrelated",
                },
                {
                    "tool": "listening_ports",
                    "status": "success",
                    "result": (
                        "Current TCP listeners:\n"
                        "- tcp 127.0.0.1:8001\n"
                        "Suggested currently-unused unprivileged TCP ports "
                        "from 8000-8999: 8002, 8003, 8004"
                    ),
                },
            ]

            instruction = agent._with_plan_recovery_evidence(
                "Correct the invalid Plan.",
                canonical_plan=json.dumps(valid_plan()),
                evidence_needed=False,
            )

        self.assertIn(
            "--- BEGIN LISTENING PORTS EVIDENCE ---",
            instruction,
        )
        self.assertIn(
            "Suggested currently-unused unprivileged TCP ports "
            "from 8000-8999: 8002, 8003, 8004",
            instruction,
        )
        self.assertNotIn(
            "from 8000-8999: 8001, 8002",
            instruction,
        )
        self.assertIn(
            "--- END LISTENING PORTS EVIDENCE ---",
            instruction,
        )

    def test_missing_port_evidence_is_gathered_by_host(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            index_path = os.path.join(directory, "index.html")

            payload = valid_plan()
            payload["objective"] = "Create a local webpage."
            payload["files"] = [index_path]
            payload["steps"] = [
                f"Create {index_path}.",
                (
                    "Run `nohup python3 -m http.server 8000 "
                    "--bind 127.0.0.1 >/dev/null 2>&1 &` so it remains "
                    "running during validation."
                ),
            ]
            payload["validation"] = [{
                "command": f"test -f {index_path}",
                "expected": "index.html exists.",
            }]

            client = mock.Mock()
            client.chat.side_effect = [{
                "role": "assistant",
                "content": fenced(payload),
            }]

            with (
                mock.patch.object(
                    core,
                    "OllamaClient",
                    return_value=client,
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_notes",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_messages",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "match_lesson_records",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "save_message",
                ),
                mock.patch.object(
                    core.memory,
                    "get_latest_plan",
                    return_value=None,
                ),
                mock.patch.object(
                    core.memory,
                    "create_plan",
                    return_value=97,
                ) as create_plan,
                mock.patch.dict(
                    core.TOOL_IMPL,
                    {
                        "listening_ports": (
                            lambda **_kwargs:
                            TEST_LISTENING_PORTS_RESULT
                        ),
                    },
                    clear=False,
                ),
                mock.patch.dict(
                    os.environ,
                    {"LIAM_HELPER_OLLAMA_URL": ""},
                    clear=False,
                ),
            ):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=True,
                    workdir=directory,
                    session_id=17,
                )
                agent.on_status = mock.Mock()
                agent.on_tool_call = mock.Mock()

                reply = agent.step(
                    "Create a complete local webpage plan, but do not "
                    "execute it."
                )

            self.assertEqual(client.chat.call_count, 1)
            agent.on_tool_call.assert_called_once_with(
                "listening_ports",
                {},
            )
            create_plan.assert_called_once()
            self.assertIn(
                "[Plan draft #97 is ready for approval.]",
                reply,
            )

            statuses = [
                call.args[0]
                for call in agent.on_status.call_args_list
                if call.args
            ]
            self.assertFalse(
                any(
                    "retrying plan evidence" in status
                    for status in statuses
                )
            )

    def test_plan_evidence_recovery_exposes_only_read_file(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "plan-ui-test.txt")
            with open(target, "w") as handle:
                handle.write("PLAN TEST\n")

            payload = valid_plan()
            payload["files"] = [target]
            payload["steps"] = [
                "Modify plan-ui-test.txt.",
            ]

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": fenced(payload),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": target},
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": fenced(payload),
                },
            ]

            with (
                mock.patch.object(
                    core,
                    "OllamaClient",
                    return_value=client,
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_notes",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_messages",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "match_lesson_records",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "save_message",
                ),
                mock.patch.object(
                    core.memory,
                    "get_latest_plan",
                    return_value=None,
                ),
                mock.patch.object(
                    core.memory,
                    "create_plan",
                    return_value=91,
                ) as create_plan,
                mock.patch.dict(
                    os.environ,
                    {"LIAM_HELPER_OLLAMA_URL": ""},
                    clear=False,
                ),
            ):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=True,
                    workdir=directory,
                    session_id=17,
                )
                agent.on_status = mock.Mock()
                agent.on_tool_call = mock.Mock()

                reply = agent.step(
                    "Create a complete plan, but do not execute it."
                )

        self.assertEqual(client.chat.call_count, 3)

        recovery_kwargs = client.chat.call_args_list[1].kwargs
        recovery_tools = recovery_kwargs.get("tools") or []
        self.assertEqual(
            [
                schema["function"]["name"]
                for schema in recovery_tools
            ],
            ["read_file"],
        )
        self.assertNotIn("response_format", recovery_kwargs)

        agent.on_tool_call.assert_called_once_with(
            "read_file",
            {"path": target},
        )
        create_plan.assert_called_once()
        self.assertIn(
            "[Plan draft #91 is ready for approval.]",
            reply,
        )

    def test_exhausted_plan_evidence_retries_do_not_save_draft(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "plan-ui-test.txt")
            with open(target, "w") as handle:
                handle.write("PLAN TEST\n")

            payload = valid_plan()
            payload["files"] = [target]
            payload["steps"] = [
                "Modify plan-ui-test.txt.",
            ]

            client = mock.Mock()
            client.chat.side_effect = [
                {
                    "role": "assistant",
                    "content": fenced(payload),
                },
                {
                    "role": "assistant",
                    "content": fenced(payload),
                },
                {
                    "role": "assistant",
                    "content": fenced(payload),
                },
            ]

            with (
                mock.patch.object(
                    core,
                    "OllamaClient",
                    return_value=client,
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_notes",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "load_recent_messages",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "match_lesson_records",
                    return_value=[],
                ),
                mock.patch.object(
                    core.memory,
                    "save_message",
                ),
                mock.patch.object(
                    core.memory,
                    "get_latest_plan",
                    return_value=None,
                ),
                mock.patch.object(
                    core.memory,
                    "create_plan",
                ) as create_plan,
                mock.patch.dict(
                    os.environ,
                    {"LIAM_HELPER_OLLAMA_URL": ""},
                    clear=False,
                ),
            ):
                agent = core.Agent(
                    channel="gui",
                    plan_mode=True,
                    workdir=directory,
                    session_id=17,
                )
                agent.on_status = mock.Mock()

                reply = agent.step(
                    "Create a complete plan, but do not execute it."
                )

        self.assertEqual(client.chat.call_count, 2)
        create_plan.assert_not_called()
        self.assertIn(
            "[Plan draft not saved after evidence recovery:",
            reply,
        )
        self.assertIn("not inspected with read_file", reply)
        self.assertNotIn("```liam-plan", reply)

        statuses = [
            call.args[0]
            for call in agent.on_status.call_args_list
            if call.args
        ]
        self.assertTrue(
            any(
                "retrying plan evidence (1/1)" in status
                for status in statuses
            )
        )
        self.assertFalse(
            any(
                "retrying plan formatting" in status
                for status in statuses
            )
        )

    @mock.patch.object(core.memory, "create_plan", return_value=41)
    @mock.patch.object(core.memory, "get_latest_plan", return_value=None)
    def test_model_plan_notice_is_removed_before_host_notice(
        self,
        get_latest_plan,
        create_plan,
    ):
        agent = core.Agent.__new__(core.Agent)
        agent.plan_mode = True
        agent.session_id = 17
        agent.messages = [{"role": "assistant", "content": "original"}]

        reply = agent._capture_plan_draft(
            fenced(valid_plan())
            + "\n\n[Plan draft #999 is ready for approval.]"
        )

        get_latest_plan.assert_called_once_with(17)
        create_plan.assert_called_once()
        self.assertNotIn(
            "[Plan draft #999 is ready for approval.]",
            reply,
        )
        self.assertEqual(
            reply.count(
                "[Plan draft #41 is ready for approval.]"
            ),
            1,
        )
        self.assertEqual(
            agent.messages[-1]["content"],
            reply,
        )


    def test_plan_format_retry_has_independent_recovery_budget(self):
        invalid = valid_plan()
        invalid.pop("non_goals")
        invalid.pop("risks")

        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": "malformed",
            },
            {
                "role": "assistant",
                "content": fenced(invalid),
            },
            {
                "role": "assistant",
                "content": fenced(valid_plan()),
            },
        ]

        with (
            mock.patch.object(
                core,
                "OllamaClient",
                return_value=client,
            ),
            mock.patch.object(
                core.memory,
                "load_recent_notes",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "load_recent_messages",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "match_lesson_records",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "save_message",
            ),
            mock.patch.object(
                core.memory,
                "get_latest_plan",
                return_value=None,
            ),
            mock.patch.object(
                core.memory,
                "create_plan",
                return_value=52,
            ) as create_plan,
            mock.patch.object(
                core.Agent,
                "_select_recovery_tool_schemas",
                return_value=[],
            ),
            mock.patch.dict(
                os.environ,
                {"LIAM_HELPER_OLLAMA_URL": ""},
                clear=False,
            ),
        ):
            agent = core.Agent(
                channel="gui",
                plan_mode=True,
                workdir="/var/www/LiamApp01",
                session_id=17,
            )
            agent.on_status = mock.Mock()

            reply = agent.step(
                "Create a complete plan, but do not execute it."
            )

        self.assertEqual(client.chat.call_count, 3)
        create_plan.assert_called_once()
        self.assertIn(
            "[Plan draft #52 is ready for approval.]",
            reply,
        )
        self.assertNotIn(
            "[Plan draft not saved:",
            reply,
        )

        statuses = [
            call.args[0]
            for call in agent.on_status.call_args_list
            if call.args
        ]
        self.assertTrue(
            any(
                "retrying plan formatting (1/2)" in status
                for status in statuses
            )
        )


    def test_prose_only_plan_is_corrected_automatically(self):
        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": (
                    "Here is the plan:\n"
                    "1. Inspect the project.\n"
                    "2. Create the webpage.\n"
                    "3. Validate it."
                ),
            },
            {
                "role": "assistant",
                "content": fenced(valid_plan()),
            },
        ]

        with (
            mock.patch.object(
                core,
                "OllamaClient",
                return_value=client,
            ),
            mock.patch.object(
                core.memory,
                "load_recent_notes",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "load_recent_messages",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "match_lesson_records",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "save_message",
            ),
            mock.patch.object(
                core.memory,
                "get_latest_plan",
                return_value=None,
            ),
            mock.patch.object(
                core.memory,
                "create_plan",
                return_value=63,
            ) as create_plan,
            mock.patch.dict(
                os.environ,
                {"LIAM_HELPER_OLLAMA_URL": ""},
                clear=False,
            ),
        ):
            agent = core.Agent(
                channel="gui",
                plan_mode=True,
                workdir="/var/www/LiamApp01",
                session_id=17,
            )
            agent.on_status = mock.Mock()

            reply = agent.step(
                "Make a plan for a local Fluxa webpage."
            )

        self.assertEqual(client.chat.call_count, 2)
        create_plan.assert_called_once()
        self.assertIn(
            "[Plan draft #63 is ready for approval.]",
            reply,
        )
        self.assertNotIn(
            "[Plan draft not saved:",
            reply,
        )

        statuses = [
            call.args[0]
            for call in agent.on_status.call_args_list
            if call.args
        ]
        self.assertTrue(
            any(
                "missing required liam-plan block" in status
                and "retrying plan formatting (1/2)" in status
                for status in statuses
            )
        )


    def test_plan_format_retry_receives_previous_invalid_answer(self):
        failed_answer = (
            "BROKEN PLAN SENTINEL\n"
            "1. Inspect the folder.\n"
            "2. Create the webpage.\n"
            "3. Validate it."
        )

        client = mock.Mock()
        client.chat.side_effect = [
            {
                "role": "assistant",
                "content": failed_answer,
            },
            {
                "role": "assistant",
                "content": fenced(valid_plan()),
            },
        ]

        with (
            mock.patch.object(
                core,
                "OllamaClient",
                return_value=client,
            ),
            mock.patch.object(
                core.memory,
                "load_recent_notes",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "load_recent_messages",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "match_lesson_records",
                return_value=[],
            ),
            mock.patch.object(
                core.memory,
                "save_message",
            ),
            mock.patch.object(
                core.memory,
                "get_latest_plan",
                return_value=None,
            ),
            mock.patch.object(
                core.memory,
                "create_plan",
                return_value=64,
            ),
            mock.patch.dict(
                os.environ,
                {"LIAM_HELPER_OLLAMA_URL": ""},
                clear=False,
            ),
        ):
            agent = core.Agent(
                channel="gui",
                plan_mode=True,
                workdir="/var/www/LiamApp01",
                session_id=18,
            )
            agent.on_status = mock.Mock()

            reply = agent.step(
                "Make a plan for a local Fluxa webpage."
            )

        self.assertEqual(client.chat.call_count, 2)

        retry_messages = client.chat.call_args_list[1].args[0]
        retry_text = retry_messages[-1]["content"]

        self.assertIn(
            "BROKEN PLAN SENTINEL",
            retry_text,
        )
        self.assertIn(
            "--- BEGIN PREVIOUS ANSWER ---",
            retry_text,
        )
        self.assertIn(
            "--- END PREVIOUS ANSWER ---",
            retry_text,
        )
        self.assertIn(
            "[Plan draft #64 is ready for approval.]",
            reply,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
