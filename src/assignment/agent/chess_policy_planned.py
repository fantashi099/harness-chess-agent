"""Bounded plan-driven executor. Planner choices are data; code owns scoring/commit."""
from copy import deepcopy
import hashlib
import json

import chess

if __package__:
    from assignment.agent import chess_policy_v3 as v3
else:
    import chess_policy_v3 as v3

POLICY_VERSION = "select-move-v4-planned-1"
MAX_PLANS = 2
MAX_QDEPTH = 16
KINDS = {"exchange", "tactical", "responses"}


def prepare(snapshot, game_id, catalog, position_id, max_nodes=6000, previous=None):
    board = v3._copy_board(snapshot)
    base = v3.select_move(board, game_id, catalog, max_nodes, variant="S")
    config = v3.profile_for(board, "R", catalog)
    mate_in_one = bool(base["top_candidates"] and base["top_candidates"][0]["primary_score"] == 100000)
    context = {"position_id": position_id, "snapshot_id": v3.identity(board, game_id),
        "game_id": game_id, "fen": board.fen(), "legal_moves": sorted(m.uci() for m in board.legal_moves),
        "top_candidates": base["top_candidates"], "retrieved": config["retrieved"],
        "rule_profile": config["profile"], "workflow_id": config["workflow_id"],
        "observed_double_attacks": v3.double_attacks(board) if not board.outcome(claim_draw=True) else [],
        "previous_feedback": {k: (previous or {}).get(k) for k in ("objective", "verification_tasks", "pv", "cutoff_reached", "primary_score")},
        "actual_reply": board.peek().uci() if board.move_stack else None,
        "budget": {"max_nodes": max_nodes, "nodes_used": base["nodes_used"], "remaining_nodes": max_nodes-base["nodes_used"],
                   "max_plans": MAX_PLANS, "max_qdepth": MAX_QDEPTH},
        "planning_required": bool(not mate_in_one and base["completed_depth"] == 2 and max_nodes > base["nodes_used"]),
        "evaluator": "Unchanged v2 material/center/check; native terminal outcome takes priority"}
    return {"context": context, "base": base, "attempts": [], "nodes_used": base["nodes_used"],
            "max_nodes": max_nodes, "selected": None}


def validate_plan(plan, context):
    required = {"position_id", "objective", "rationale", "candidate_moves", "workflow_id", "verification_tasks"}
    if not isinstance(plan, dict) or set(plan) != required or plan["position_id"] != context["position_id"]:
        raise ValueError("Malformed or stale plan")
    for key, limit in (("objective", 320), ("rationale", 2000)):
        if not isinstance(plan[key], str) or not 1 <= len(plan[key]) <= limit:
            raise ValueError(f"Invalid {key}")
    candidates = plan["candidate_moves"]
    if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 5
            or any(not isinstance(m, str) or m not in context["legal_moves"] for m in candidates)
            or len(set(candidates)) != len(candidates)):
        raise ValueError("Candidates must be 1–5 unique legal White moves")
    if plan["workflow_id"] not in (None, context["workflow_id"]):
        raise ValueError("Workflow was not retrieved/admitted")
    tasks = plan["verification_tasks"]
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 2:
        raise ValueError("One or two bounded tasks are required")
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"id", "kind", "depends_on", "max_qdepth"}:
            raise ValueError("Malformed verification task")
        if (not isinstance(task["id"], str) or not 1 <= len(task["id"]) <= 64 or task["id"] in seen
                or not isinstance(task["kind"], str) or task["kind"] not in KINDS or type(task["max_qdepth"]) is not int
                or not 0 <= task["max_qdepth"] <= MAX_QDEPTH
                or not isinstance(task["depends_on"], list)
                or any(not isinstance(d, str) or d not in seen for d in task["depends_on"])):
            raise ValueError("Invalid task kind, dependency or depth")
        seen.add(task["id"])
    return deepcopy(plan)


def rule_plan(context, feedback=None):
    depth = context["rule_profile"]["max_qdepth"]
    if feedback and feedback.get("status") == "unresolved":
        depth = MAX_QDEPTH
    return {"position_id": context["position_id"], "objective": "Verify a common candidate scope before committing one move",
        "rationale": "Matched rule baseline: top-five shallow candidates, conditional fork responses, bounded depth escalation",
        "candidate_moves": [r["move"] for r in context["top_candidates"]], "workflow_id": context["workflow_id"],
        "verification_tasks": [{"id": "tactical", "kind": "tactical", "depends_on": [], "max_qdepth": depth}]}


def execute_plan(snapshot, game_id, session, plan):
    board = v3._copy_board(snapshot)
    context = session["context"]
    if context["snapshot_id"] != v3.identity(board, game_id):
        raise ValueError("Stale snapshot/game/history")
    if not context["planning_required"]:
        raise ValueError("Opening, terminal or exhausted-budget preparation needs no plan")
    plan = validate_plan(plan, context)
    plan_id = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    for old in session["attempts"]:
        if old["plan_id"] == plan_id:
            return deepcopy(old)  # Retrying verification must not consume nodes again.
    if session["selected"] is not None:
        raise ValueError("Verification objective already completed")
    if len(session["attempts"]) >= MAX_PLANS:
        raise ValueError("Plan attempt limit reached")
    trace = {"plan_id": plan_id, "plan": plan, "status": "unresolved", "score": None,
             "nodes": 0, "invocations": [], "completed_candidates": [], "cutoff_reached": False,
             "scope": "Every legal Black reply for the declared candidate subset; task scopes and last-task scores only",
             "candidates_omitted": sorted(set(context["legal_moves"])-set(plan["candidate_moves"]))}
    started_nodes = session["nodes_used"]

    def finish(status, **extra):
        trace.update(status=status, nodes=session["nodes_used"]-started_nodes,
                     total_nodes_used=session["nodes_used"], remaining_nodes=session["max_nodes"]-session["nodes_used"], **extra)
        session["attempts"].append(deepcopy(trace))
        if status == "completed":
            session["selected"] = deepcopy(trace["selected"])
        return trace

    root = dict(v3._state(board), fen=board.fen())
    ranked = []
    for candidate in plan["candidate_moves"]:
        move = chess.Move.from_uci(candidate)
        board.push(move)
        try:
            if board.outcome(claim_draw=True):
                value, pv = v3.evaluate(v3._state(board)), [candidate]
            else:
                value, pv = float("inf"), []
                replies = sorted(board.legal_moves, key=lambda m: m.uci())
                for reply_index, reply in enumerate(replies):
                    board.push(reply)
                    try:
                        for task in plan["verification_tasks"]:
                            remaining = session["max_nodes"]-session["nodes_used"]
                            if remaining < 1:
                                return finish("unresolved", cutoff_reached=True, stop_reason="max_nodes",
                                              pending_candidate=candidate, pending_reply=reply.uci(), pending_task=task["id"])
                            if task["kind"] == "responses" or (task["kind"] == "tactical" and v3.double_attacks(board)):
                                helper_name = "verify_responses"
                                result = v3.verify_responses(board, remaining, task["max_qdepth"], max_escalations=1)
                            elif plan["workflow_id"] == v3.WORKFLOW and not v3._moves(board) and not board.outcome(claim_draw=True):
                                helper_name = "workflow_skip"
                                result = {"nodes": 1, "score": v3.evaluate(v3._state(board)), "continuation": [],
                                          "cutoff_reached": False, "stop_reason": "workflow_not_relevant"}
                            else:
                                helper_name = "verify_exchange"
                                result = v3.verify_exchange(board, budget={"max_nodes": remaining, "max_qdepth": task["max_qdepth"]})
                            session["nodes_used"] += result["nodes"]
                            trace["invocations"].append({"candidate": candidate, "reply": reply.uci(), "task": task,
                                "helper": helper_name, "budget_passed": remaining, "result": result})
                            if result["cutoff_reached"]:
                                return finish("unresolved", cutoff_reached=True, stop_reason=result["stop_reason"],
                                    pending_candidate=candidate, pending_reply=reply.uci(), pending_task=task["id"],
                                    cutoff_path=result.get("cutoff_path", result.get("continuation", [])),
                                    replies_unvisited_current_candidate=len(replies)-reply_index-1)
                    finally:
                        board.pop()
                    if result["score"] < value:
                        value, pv = result["score"], [candidate, reply.uci()] + result["continuation"]
        finally:
            board.pop()
        features, key = v3.secondary(root, candidate, session["base"]["recent_white_moves"])
        row = {"move": candidate, "primary_score": value, "pv": pv, "secondary": features, "secondary_key": key}
        ranked.append(row)
        trace["completed_candidates"].append(row)
    ranked.sort(key=lambda r: (-r["primary_score"], tuple(-v for v in r["secondary_key"]), r["move"]))
    return finish("completed", selected=ranked[0], score=ranked[0]["primary_score"], stop_reason="declared_scope_completed")


def decision_from(session, planner, planner_records=None):
    base = deepcopy(session["base"])
    selected = session["selected"]
    if selected is None and base["top_candidates"]:
        selected = base["top_candidates"][0]
    if selected is not None:
        base.update(chosen_move=selected["move"], primary_score=selected["primary_score"],
                    pv=selected["pv"], secondary=selected["secondary"])
    base.update(policy_version=POLICY_VERSION, planner=planner, planning_context=session["context"],
        execution_attempts=deepcopy(session["attempts"]), planner_records=planner_records or [],
        preparation_nodes=session["base"]["nodes_used"], nodes_used=session["nodes_used"], simulator_calls=session["nodes_used"],
        cutoff_reached=session["attempts"][-1]["cutoff_reached"] if session["attempts"] else base["cutoff_reached"],
        reason="completed_plan_verification" if session["selected"] else "completed_preparation_fallback" if base["completed_depth"] == 2 else base["reason"],
        plan=session["attempts"][-1]["plan"] if session["attempts"] else None,
        score_scope="Last completed common plan scope" if session["selected"] else "Preparation fallback scope")
    return base


def feedback_summary(trace):
    return {k: trace.get(k) for k in ("status", "plan", "score", "stop_reason", "cutoff_reached", "cutoff_path",
        "nodes", "remaining_nodes", "pending_candidate", "pending_reply", "pending_task", "completed_candidates")}
