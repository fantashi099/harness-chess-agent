"""Constrain action shape without selecting a move or relaxing executor checks."""
import pytest

from assignment.agent.chess_harness import action_response_format
from assignment.agent.chess_policy_harness import validate_action

CONTEXT = {"legal_moves": ["e2e4", "d2d4"], "budget": {"max_nodes": 6000, "max_depth": 4}}


@pytest.mark.parametrize("allowed", [["analyze"], ["commit"], ["analyze", "commit"]])
def test_schema_exposes_exactly_the_allowed_actions(allowed):
    result = action_response_format(CONTEXT, {"allowed_actions": allowed})
    assert result["type"] == "json_schema" and result["json_schema"]["strict"]
    schema = result["json_schema"]["schema"]
    branches = schema.get("anyOf", [schema])
    assert {b["properties"]["action"]["enum"][0] for b in branches} == set(allowed)
    for branch in branches:
        assert branch["properties"]["action"]["type"] == "string"
        assert branch["additionalProperties"] is False
        assert set(branch["required"]) == set(branch["properties"])
        if branch["properties"]["action"]["enum"] == ["analyze"]:
            assert branch["properties"]["max_nodes"]["maximum"] == 6000
            assert branch["properties"]["max_depth"]["maximum"] == 4
            assert branch["properties"]["candidate_moves"]["items"]["enum"] == CONTEXT["legal_moves"]


def test_optional_commit_fields_remain_optional_and_do_not_choose_the_move():
    schema = action_response_format(CONTEXT, {"allowed_actions": ["commit"]})["json_schema"]["schema"]
    combinations = set()
    for branch in schema["anyOf"]:
        properties = branch["properties"]
        assert properties["move"]["enum"] == CONTEXT["legal_moves"]
        optional = frozenset(properties) & {"continuation", "accept_risk"}
        combinations.add(optional)
        action = {"action": "commit", "objective": "Develop", "rationale": "Use verified evidence", "move": "d2d4"}
        if "continuation" in optional:
            action["continuation"] = ["d2d4"]
        if "accept_risk" in optional:
            action["accept_risk"] = False
        assert validate_action(action, CONTEXT) == action
    assert combinations == {frozenset(), frozenset({"continuation"}),
                            frozenset({"accept_risk"}), frozenset({"continuation", "accept_risk"})}


@pytest.mark.parametrize("value", [None, True, 9, ["analyze"], "", ".analyze", {"action": "analyze"}])
def test_malformed_provider_actions_still_fail_closed(value):
    with pytest.raises(ValueError, match="Action must be analyze or commit"):
        validate_action({"action": value}, CONTEXT)


def test_no_allowed_action_never_creates_an_unrestricted_schema():
    with pytest.raises(ValueError, match="No supported actions"):
        action_response_format(CONTEXT, {"allowed_actions": []})
