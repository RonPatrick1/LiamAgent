import json
import unittest
from unittest import mock

from agent import core


class PlanSemanticGateTests(unittest.TestCase):
    @staticmethod
    def _agent():
        agent = core.Agent.__new__(core.Agent)
        agent.on_status = mock.Mock()
        return agent

    def test_legacy_positive_match_avoids_classifier_call(self):
        agent = self._agent()
        agent._helper_chat = mock.Mock()

        result = agent._plan_draft_required_for_request(
            "Create a local webpage."
        )

        self.assertTrue(result)
        agent._helper_chat.assert_not_called()

    def test_indirect_project_improvement_request_requires_plan(self):
        agent = self._agent()
        agent._helper_chat = mock.Mock(return_value={
            "content": json.dumps({
                "requires_plan": True,
            }),
        })

        result = agent._plan_draft_required_for_request(
            "Look through this project and tell me how you would improve it."
        )

        self.assertTrue(result)
        self.assertIs(
            agent._helper_chat.call_args.kwargs["response_format"],
            core.PLAN_REQUEST_CLASSIFICATION_SCHEMA,
        )

    def test_pure_explanation_does_not_require_plan(self):
        agent = self._agent()
        agent._helper_chat = mock.Mock(return_value={
            "content": json.dumps({
                "requires_plan": False,
            }),
        })

        result = agent._plan_draft_required_for_request(
            "Explain how local network ports work."
        )

        self.assertFalse(result)

    def test_invalid_classifier_output_requires_plan_conservatively(self):
        agent = self._agent()
        agent._helper_chat = mock.Mock(return_value={
            "content": "not valid json",
        })

        result = agent._plan_draft_required_for_request(
            "Take a look around and tell me what you would do next."
        )

        self.assertTrue(result)
        agent.on_status.assert_called_once()
        self.assertIn(
            "requiring a plan conservatively",
            agent.on_status.call_args.args[0],
        )


if __name__ == "__main__":
    unittest.main()
