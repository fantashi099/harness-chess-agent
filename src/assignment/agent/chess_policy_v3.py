"""Opt-in bounded rule planner/executor with frozen conditional memory.

The two-ply fallback uses v2 evaluation/tie breaks, but retains Board history.
Only completed, equally scoped shortlist scores are compared. No live commit.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from time import monotonic

import chess

if __package__:
    from assignment.agent.chess_policy import INITIAL_FEN, VALUES, evaluate, position_id, secondary
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path
    # CLI entry points do not put the repository root on sys.path.
    spec = spec_from_file_location("exchange_verifier", Path(__file__).resolve().parents[3] / "experiments/chess/verify_exchange.py")
    helper = module_from_spec(spec)
    spec.loader.exec_module(helper)
    _copy_board, _moves, _state, verify_exchange = helper._copy_board, helper._moves, helper._state, helper.verify_exchange
else:  # Explicitly copied sandbox files.
    from chess_policy import INITIAL_FEN, VALUES, evaluate, position_id, secondary
    from verify_exchange import _copy_board, _moves, _state, verify_exchange

POLICY_VERSION = "select-move-v3-bounded-1"
DEFAULT_NODES = 6000
WORKFLOW = "verify-tactical-exchange"


def identity(board, game_id):
    return position_id({"fen": board.fen(), "history": [m.uci() for m in board.move_stack],
                        "policy_game_id": game_id})


def double_attacks(board):
    """Observed opponent geometry; evidence of targets, never forced loss."""
    found = []
    # ponytail: geometric attacks include pinned pieces; add legal threat search if needed.
    for square, piece in sorted(board.piece_map().items()):
        if piece.color == board.turn:
            continue
        targets = [chess.square_name(s) for s in sorted(board.attacks(square))
                   if (p := board.piece_at(s)) and p.color == board.turn
                   and p.piece_type not in {chess.PAWN, chess.KING}]
        if len(targets) >= 2:
            found.append({"attacker": chess.square_name(square), "piece": piece.symbol(),
                          "targets": targets})
    return found


def retrieve(board, catalog):
    trigger = ("in_check" if board.is_check() else
               "legal_tactical_move" if _moves(board) else None)
    if trigger is None or board.outcome(claim_draw=True) is not None:
        return []
    # ponytail: two rule-selected entries; embeddings only if this small catalog grows.
    return [dict(item, retrieval_reason=trigger) for item in catalog.get("entries", [])
            if item.get("status") == "active" and trigger in item.get("triggers", [])][:2]


def profile_for(board, variant, catalog):
    started = monotonic()
    if variant not in {"S", "P", "R"}:
        raise ValueError("variant must be S, P, or R")
    profile = {"max_qdepth": 2 if variant == "S" else 4,
               "extend_observed_double_attack": variant != "S", "max_escalations": 1}
    before = deepcopy(profile)
    retrieval_started = monotonic()
    items = retrieve(board, catalog) if variant == "R" else []
    retrieval_seconds = monotonic() - retrieval_started
    for item in items:
        action = item.get("corrective_action", {})
        if "max_qdepth" in action:
            depth = action["max_qdepth"]
            if type(depth) is not int or not 0 <= depth <= 64:
                raise ValueError("Invalid memory depth")
            profile["max_qdepth"] = max(profile["max_qdepth"], depth)
    workflow = next((x["id"] for x in items if x.get("kind") == "procedural"), None)
    if workflow is not None and workflow != WORKFLOW:
        raise ValueError("Unadmitted workflow")
    return {"profile_before_retrieval": before, "profile": profile, "retrieved": items,
            "workflow_id": workflow, "retrieval_seconds": retrieval_seconds,
            "planner_seconds": monotonic() - started}


def verify_responses(snapshot, max_nodes, max_qdepth, max_escalations=1):
    """One bounded decomposition: targets -> legal responses -> continuations."""
    board = _copy_board(snapshot)
    if (type(max_nodes) is not int or max_nodes < 1 or type(max_escalations) is not int or max_escalations < 0
            or type(max_qdepth) is not int or not 0 <= max_qdepth <= 64):
        raise ValueError("Invalid decomposition budget")
    result = {"nodes": 1, "tasks": [], "score": None, "continuation": [],
              "cutoff_reached": False, "max_escalations": max_escalations,
              "escalations_used": 0, "scope": "All legal first responses, followed by bounded quiescence."}
    outcome = board.outcome(claim_draw=True)
    if outcome:
        return dict(result, score=evaluate(_state(board)), stop_reason="terminal")
    targets = double_attacks(board)
    result["tasks"].append({"id": "targets", "depends_on": [], "evidence": targets})
    if max_escalations == 0:
        return dict(result, cutoff_reached=True, stop_reason="max_escalations")
    result["escalations_used"] = 1
    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    result["tasks"].append({"id": "responses", "depends_on": ["targets"],
                            "evidence": [m.uci() for m in moves]})
    checked = []
    result["tasks"].append({"id": "continuations", "depends_on": ["responses"], "evidence": checked})
    best = None
    for move in moves:
        remaining = max_nodes - result["nodes"]
        if remaining < 1:
            return dict(result, cutoff_reached=True, stop_reason="max_nodes")
        board.push(move)
        try:
            search = verify_exchange(board, budget={"max_nodes": remaining, "max_qdepth": max_qdepth})
        finally:
            board.pop()
        result["nodes"] += search["nodes"]
        checked.append({"move": move.uci(), "result": search})
        if search["cutoff_reached"]:
            return dict(result, cutoff_reached=True, stop_reason=search["stop_reason"])
        score = search["score"]
        if best is None or (score > best if board.turn else score < best):
            best = score
            result["continuation"] = [move.uci()] + search["continuation"]
    return dict(result, score=best, stop_reason="all_responses_checked")


def replan(board, game_id, previous, observation):
    threats = double_attacks(board) if board.outcome(claim_draw=True) is None else []
    cutoff = bool(observation.get("cutoff_reached"))
    revise = bool(threats or cutoff)
    objective = ("Verify legal responses to the observed double attack" if threats else
                 "Resolve uncertainty within the remaining bounded search" if cutoff else
                 previous.get("objective", "Evaluate unfinished tactical exchanges"))
    tasks = (["targets", "responses", "continuations"] if threats else
             ["bounded_verification"] if cutoff else ["exchange"])
    new_id = identity(board, game_id)
    plan_id = hashlib.sha256(json.dumps([new_id, objective, tasks]).encode()).hexdigest()[:16]
    reply_changed = (observation.get("actual_reply") is not None and observation.get("predicted_reply") is not None
                     and observation["actual_reply"] != observation["predicted_reply"])
    return {"old_plan_id": previous.get("plan_id"), "plan_id": plan_id,
            "new_position_id": new_id, "objective": objective, "observation": observation,
            "trigger": "observed_double_attack" if threats else "verifier_cutoff" if cutoff else "reply_differs_from_pv" if reply_changed else "root_updated",
            "keep/revise": "revise" if revise else "keep", "targets": threats,
            "tasks_added": [t for t in tasks if t not in previous.get("tasks", [])],
            "tasks_removed": [t for t in previous.get("tasks", []) if t not in tasks], "tasks": tasks}


def select_move(snapshot, game_id, catalog, max_nodes=DEFAULT_NODES, variant="R", previous=None):
    board = _copy_board(snapshot)
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    if not isinstance(game_id, str) or not game_id or not isinstance(catalog, dict):
        raise ValueError("A game ID and catalog object are required")
    config = profile_for(board, variant, catalog)
    nodes = 1
    root = dict(_state(board), fen=board.fen())
    recent = [m.uci() for i, m in enumerate(board.move_stack)
              if (board.root().turn if i % 2 == 0 else not board.root().turn) == chess.WHITE][-4:]
    observed = {"actual_reply": board.peek().uci() if board.move_stack else None,
                "predicted_reply": previous.get("pv", [None, None])[1] if previous and len(previous.get("pv", [])) > 1 else None,
                "cutoff_reached": previous.get("cutoff_reached", False) if previous else False}
    active = replan(board, game_id, previous or {"objective": "Evaluate unfinished tactical exchanges", "tasks": ["exchange"]}, observed)
    plan = dict(config, **active, position_id=identity(board, game_id), game_id=game_id,
                budget={"max_nodes": max_nodes}, variant=variant)
    common = {"policy_version": POLICY_VERSION, "root_fen": board.fen(), "plan": plan,
              "repetition_history_available": True, "simulator_backend": "immutable_board_stack",
              "primary_ties": [], "top_candidates": [], "pv": [], "secondary": {},
              "cutoff_reached": False, "verification": [], "refinement_candidates_skipped": 0,
              "refinement_replies_skipped": 0, "refinement_candidates_unfinished": 0, "nodes_used": nodes, "simulator_calls": nodes,
              "completed_depth": 0, "search_performed": False, "recent_white_moves": recent}

    def finish(**changes):
        return dict(common, nodes_used=nodes, simulator_calls=nodes, **changes)

    if board.outcome(claim_draw=True):
        return finish(chosen_move=None, primary_score=evaluate(root), reason="terminal_position")
    if board.turn != chess.WHITE:
        raise ValueError("v3 selects only White moves")
    moves = sorted(board.legal_moves, key=lambda m: m.uci())
    if board.fen() == INITIAL_FEN:
        return finish(chosen_move="e2e4", primary_score=None, pv=["e2e4"], reason="opening_preference")
    scored = []
    try:
        for move in moves:
            if nodes >= max_nodes:
                raise RuntimeError("max_nodes")
            board.push(move)
            nodes += 1
            try:
                if board.outcome(claim_draw=True):
                    score, pv = evaluate(_state(board)), [move.uci()]
                else:
                    score, pv = float("inf"), []
                    for reply in sorted(board.legal_moves, key=lambda m: m.uci()):
                        if nodes >= max_nodes:
                            raise RuntimeError("max_nodes")
                        board.push(reply)
                        nodes += 1
                        try:
                            value = evaluate(_state(board))
                        finally:
                            board.pop()
                        if value < score:
                            score, pv = value, [move.uci(), reply.uci()]
            finally:
                board.pop()
            features, key = secondary(root, move.uci(), recent)
            scored.append({"move": move.uci(), "primary_score": score, "pv": pv,
                           "secondary": features, "secondary_key": key})
    except RuntimeError:
        return finish(chosen_move=moves[0].uci(), primary_score=None, reason="legal_fallback_unsearched",
                      cutoff_reached=True, candidates_evaluated=len(scored),
                      baseline_candidates_skipped=len(moves)-len(scored))
    ranked = sorted(scored, key=lambda r: (-r["primary_score"], tuple(-v for v in r["secondary_key"]), r["move"]))
    best = ranked[0]
    shortlist = ranked[:5]
    common.update(search_performed=True, completed_depth=2, candidates_evaluated=len(scored),
                  top_candidates=shortlist, baseline_candidates_skipped=0,
                  refinement_candidates_skipped=len(ranked)-len(shortlist))
    refined = []
    # ponytail: only top five shallow candidates are refined; widen if measured omissions matter.
    for candidate in shortlist:
        move = chess.Move.from_uci(candidate["move"])
        board.push(move)
        try:
            replies = sorted(board.legal_moves, key=lambda m: m.uci())
            if board.outcome(claim_draw=True):
                refined.append(candidate)
                continue
            worst, line = float("inf"), []
            for index, reply in enumerate(replies):
                board.push(reply)
                try:
                    if nodes >= max_nodes:
                        search = {"nodes": 0, "score": None, "cutoff_reached": True, "stop_reason": "max_nodes"}
                    else:
                        profile = config["profile"]
                        if (profile["extend_observed_double_attack"] and board.outcome(claim_draw=True) is None
                                and double_attacks(board)):
                            search = verify_responses(board, max_nodes-nodes, profile["max_qdepth"], profile["max_escalations"])
                        elif (config["workflow_id"] == WORKFLOW and not _moves(board)
                              and board.outcome(claim_draw=True) is None):
                            # Admitted workflow: skip the verifier when its leaf trigger is absent.
                            search = {"nodes": 1, "score": evaluate(_state(board)), "continuation": [],
                                      "cutoff_reached": False, "stop_reason": "workflow_not_relevant",
                                      "verifier_calls": 0}
                        else:
                            search = verify_exchange(board, budget={"max_nodes": max_nodes-nodes,
                                                                      "max_qdepth": profile["max_qdepth"]})
                finally:
                    board.pop()
                nodes += search["nodes"]
                common["verification"].append({"candidate": move.uci(), "reply": reply.uci(),
                    "workflow_id": config["workflow_id"], "profile": config["profile"], "result": search})
                if search["cutoff_reached"]:
                    common["cutoff_reached"] = True
                    common["refinement_replies_skipped"] = len(replies)-index-1
                    common["refinement_candidates_unfinished"] = len(shortlist)-len(refined)
                    return finish(chosen_move=best["move"], primary_score=best["primary_score"], pv=best["pv"],
                                  secondary=best["secondary"], reason="completed_two_ply_fallback",
                                  primary_ties=[r for r in scored if r["primary_score"] == best["primary_score"]])
                if search["score"] < worst:
                    worst, line = search["score"], [move.uci(), reply.uci()] + search["continuation"]
            refined.append(dict(candidate, primary_score=worst, pv=line))
        finally:
            board.pop()
    refined.sort(key=lambda r: (-r["primary_score"], tuple(-v for v in r["secondary_key"]), r["move"]))
    chosen = refined[0]
    return finish(chosen_move=chosen["move"], primary_score=chosen["primary_score"], pv=chosen["pv"],
                  secondary=chosen["secondary"], reason="completed_shortlist_verification",
                  completed_depth=2, score_scope="Two-ply top-five shortlist plus declared extensions",
                  primary_ties=[r for r in refined if r["primary_score"] == chosen["primary_score"]])
