"""Host-owned action contracts and structured tool-event matching.

Models may propose contracts, but only this module validates them against the
real tool registry and decides whether an observed tool event satisfies one.
"""

from copy import deepcopy


ACTION_CONTRACT_PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "actionable",
        "operation",
        "required_capability",
        "completion_mode",
        "preferred_tool",
        "targets",
        "constraints",
        "needs_clarification",
    ],
    "properties": {
        "actionable": {"type": "boolean"},
        "operation": {"type": ["string", "null"]},
        "required_capability": {"type": ["string", "null"]},
        "completion_mode": {"type": ["string", "null"]},
        "preferred_tool": {"type": ["string", "null"]},
        "targets": {"type": "object"},
        "constraints": {"type": "object"},
        "needs_clarification": {"type": "boolean"},
    },
}


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _definition_targets(definition, args):
    return {
        field: deepcopy(args[field])
        for field in definition.get("target_fields", ())
        if field in args and args[field] not in (None, "", [], {})
    }


def build_tool_event(
    tool_name,
    args,
    result,
    outcome,
    tool_definitions,
    contract_id=None,
):
    """Enrich an existing classified outcome with registry-owned metadata."""
    definition = tool_definitions.get(tool_name)
    event = dict(outcome)

    if definition is None:
        event.update({
            "capabilities": [],
            "effect_kind": None,
            "completion_modes": [],
            "targets": {},
            "evidence": {"arguments": deepcopy(args)},
        })
    else:
        event.update({
            "capabilities": list(definition["capabilities"]),
            "effect_kind": definition["effect_kind"],
            "completion_modes": list(definition["completion_modes"]),
            "targets": _definition_targets(definition, args),
            "evidence": {"arguments": deepcopy(args)},
        })

    if contract_id is not None:
        event["contract_id"] = contract_id

    return event


def validate_contract_proposal(
    proposal,
    tool_definitions,
    offered_tool_names=None,
):
    """Validate a model proposal against the host's actual tool registry.

    Returns None for a verified non-action. Raises ValueError for malformed or
    unsupported action proposals. The model never gets to invent capabilities,
    completion modes, or tools.
    """
    if not isinstance(proposal, dict):
        raise ValueError("Contract proposal must be an object.")

    if proposal.get("actionable") is False:
        return None

    if proposal.get("actionable") is not True:
        raise ValueError("Contract proposal must declare actionable true or false.")

    operation = proposal.get("operation")
    capability = proposal.get("required_capability")
    completion_mode = proposal.get("completion_mode")
    preferred_tool = proposal.get("preferred_tool")
    targets = proposal.get("targets")
    constraints = proposal.get("constraints")
    needs_clarification = proposal.get("needs_clarification")

    if not _nonempty_string(operation):
        raise ValueError("Action contracts require a non-empty operation.")
    if not _nonempty_string(capability):
        raise ValueError("Action contracts require a capability.")
    if not _nonempty_string(completion_mode):
        raise ValueError("Action contracts require a completion mode.")
    if preferred_tool is not None and not _nonempty_string(preferred_tool):
        raise ValueError("preferred_tool must be null or a non-empty tool name.")
    if not isinstance(targets, dict):
        raise ValueError("Action contract targets must be an object.")
    if not isinstance(constraints, dict):
        raise ValueError("Action contract constraints must be an object.")
    if not isinstance(needs_clarification, bool):
        raise ValueError("needs_clarification must be true or false.")

    offered = (
        set(tool_definitions)
        if offered_tool_names is None
        else set(offered_tool_names)
    )

    if constraints:
        raise ValueError(
            "Action contract constraints are not supported until a tool "
            "definition declares structured constraint evidence."
        )

    if preferred_tool is not None:
        definition = tool_definitions.get(preferred_tool)
        if definition is None:
            raise ValueError(f"Preferred tool {preferred_tool!r} does not exist.")
        if preferred_tool not in offered:
            raise ValueError(
                f"Preferred tool {preferred_tool!r} is not offered in this conversation."
            )
        if capability not in definition["capabilities"]:
            raise ValueError(
                f"Preferred tool {preferred_tool!r} does not provide "
                f"capability {capability!r}."
            )
        if completion_mode not in definition["completion_modes"]:
            raise ValueError(
                f"Preferred tool {preferred_tool!r} does not support "
                f"completion mode {completion_mode!r}."
            )

    eligible = {
        name: definition
        for name, definition in tool_definitions.items()
        if name in offered
        and capability in definition["capabilities"]
        and completion_mode in definition["completion_modes"]
    }

    if not eligible:
        raise ValueError(
            f"No offered tool provides capability {capability!r} "
            f"with completion mode {completion_mode!r}."
        )

    target_definitions = (
        {preferred_tool: tool_definitions[preferred_tool]}
        if preferred_tool is not None
        else eligible
    )
    supported_target_fields = {
        field
        for definition in target_definitions.values()
        for field in definition.get("target_fields", ())
    }
    unsupported_targets = sorted(
        set(targets) - supported_target_fields
    )
    if unsupported_targets:
        raise ValueError(
            "Unsupported contract target fields: "
            + ", ".join(unsupported_targets)
        )

    return {
        "operation": operation.strip(),
        "required_capability": capability.strip(),
        "completion_mode": completion_mode.strip(),
        "preferred_tool": preferred_tool.strip() if preferred_tool else None,
        "targets": deepcopy(targets),
        "constraints": deepcopy(constraints),
        "status": (
            "needs_clarification"
            if needs_clarification
            else "pending"
        ),
    }


def _value_satisfies(expected, actual):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _value_satisfies(value, actual[key])
            for key, value in expected.items()
        )

    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return False
        return all(
            any(_value_satisfies(item, candidate) for candidate in actual)
            for item in expected
        )

    if isinstance(actual, (list, tuple)):
        return any(_value_satisfies(expected, candidate) for candidate in actual)

    return expected == actual


def event_satisfies_contract(contract, event):
    """Return whether one machine-observed event completes one contract."""
    if not isinstance(contract, dict) or not isinstance(event, dict):
        return False
    if event.get("status") != "success":
        return False

    capability = contract.get("required_capability")
    if capability not in event.get("capabilities", ()):
        return False

    completion_mode = contract.get("completion_mode")
    if completion_mode != event.get("effect_kind"):
        return False

    preferred_tool = contract.get("preferred_tool")
    if preferred_tool and event.get("tool") != preferred_tool:
        return False

    expected_targets = contract.get("targets") or {}
    if not _value_satisfies(expected_targets, event.get("targets") or {}):
        return False

    expected_constraints = contract.get("constraints") or {}
    if not _value_satisfies(
        expected_constraints,
        event.get("evidence") or {},
    ):
        return False

    return True


class PersistentActionContractStore:
    """MySQL-backed contract lifecycle adapter used by production Agents."""

    def create(self, session_id, source_text, contract):
        from . import memory
        return memory.create_action_contract(
            session_id,
            source_text,
            contract,
        )

    def get_active(self, session_id):
        from . import memory
        return memory.get_active_action_contract(session_id)

    def get(self, contract_id):
        from . import memory
        return memory.get_action_contract(contract_id)

    def record_event(self, session_id, event, contract_id=None):
        from . import memory
        return memory.record_action_tool_event(
            session_id,
            event,
            contract_id=contract_id,
        )

    def transition(
        self,
        contract_id,
        expected_status,
        new_status,
        matched_event_id=None,
        failure_reason=None,
    ):
        from . import memory
        return memory.transition_action_contract(
            contract_id,
            expected_status,
            new_status,
            matched_event_id=matched_event_id,
            failure_reason=failure_reason,
        )
