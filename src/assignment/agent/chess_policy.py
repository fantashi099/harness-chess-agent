"""Deterministic v2 policy; shared by offline checks and the sandbox server."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

POLICY_VERSION = "select-move-v2-fixed-1"
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}
CENTERS = {"d4", "e4", "d5", "e5"}
HOMES = {"N": {"b1", "g1"}, "B": {"c1", "f1"},
         "R": {"a1", "h1"}, "Q": {"d1"}}
FEATURE_NAMES = ("castle", "develop_minor", "advance_center",
                 "early_major_penalty", "return_home_penalty", "recent_move_penalty")


def position_id(state: dict[str, Any]) -> str:
    """Include history so a reset or repetition context invalidates a request."""
    payload = {"fen": state["fen"], "history": state["history"],
               "game_id": state.get("policy_game_id")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def evaluate(state: dict[str, Any]) -> int:
    if state["game_over"]:
        winner = state["winner"]
        return 100_000 if winner == "white" else -100_000 if winner == "black" else 0
    score = 0
    for square, piece in state["squares"].items():
        if piece == ".":
            continue
        value = VALUES[piece.upper()] + (12 if square in CENTERS else 0)
        score += value if piece.isupper() else -value
    if state["in_check"]:
        score += 25 if state["turn"] == "black" else -25
    return score


def secondary(root: dict[str, Any], move: str, recent: Sequence[str]) -> tuple[dict, tuple]:
    source, target = move[:2], move[2:4]
    piece = root["squares"].get(source, ".")
    castle = int(piece == "K" and move in {"e1g1", "e1c1"})
    develop = int(piece in {"N", "B"} and source in HOMES[piece]
                  and target not in HOMES[piece])
    center = int(piece == "P" and move in {"d2d3", "d2d4", "e2e3", "e2e4"})
    early = int(int(root["fen"].split()[5]) <= 10
                and piece in {"K", "Q", "R"} and not castle)
    homes = HOMES.get(piece, set())
    returning = int(source not in homes and target in homes)
    repeated = int(move in recent)
    reversed_move = int(any(old[:2] == target and old[2:4] == source for old in recent))
    features = (castle, develop, center, early, returning, repeated + reversed_move)
    key = features[:3] + tuple(-value for value in features[3:])
    return dict(zip(FEATURE_NAMES, features)), key


def select_move(
    fen: str,
    simulate_move: Callable[..., dict[str, Any]],
    recent_white_moves: Sequence[str] = (),
) -> dict[str, Any]:
    """Complete two-ply minimax before returning; never mutate the live board.

    This preserves the text skill's evaluation, opening and tie-breaks.
    Simulator failures abort the decision: there is no unsearched fallback.
    """
    calls = 0

    def simulate(position: str, move: str | None = None) -> dict:
        nonlocal calls
        calls += 1
        return simulate_move(position, move)

    root = simulate(fen)
    recent = list(recent_white_moves[-4:])
    common = {
        "policy_version": POLICY_VERSION, "root_fen": root["fen"],
        "recent_white_moves": recent, "repetition_history_available": False,
        "simulator_backend": "in_process_simulate_position",
    }
    if root["game_over"]:
        return dict(common, chosen_move=None, primary_score=evaluate(root),
                    search_performed=False, completed_depth=0, simulator_calls=calls,
                    primary_ties=[], top_candidates=[], pv=[], reason="terminal_position")
    if root["turn"] != "white":
        raise ValueError("The fixed v2 policy selects moves only for White.")
    candidates = sorted(root["legal_moves"])
    if not candidates:
        raise ValueError("Nonterminal White root has no legal moves.")
    if root["fen"] == INITIAL_FEN:
        best = next(move for move in ("e2e4", "d2d4", "c2c4", "g1f3")
                    if move in candidates)
        features, _ = secondary(root, best, recent)
        return dict(common, chosen_move=best, primary_score=None, secondary=features,
                    pv=[best], primary_ties=[], top_candidates=[], search_performed=False,
                    completed_depth=0, simulator_calls=calls, reason="opening_preference")
    scored = []
    for move in candidates:
        after = simulate(root["fen"], move)
        worst_reply = None
        if after["game_over"]:
            primary = evaluate(after)
        else:
            replies = sorted(after["legal_moves"])
            if not replies:
                raise ValueError("Nonterminal child has no Black replies.")
            primary = float("inf")
            for reply in replies:
                score = evaluate(simulate(after["fen"], reply))
                if score < primary:
                    primary, worst_reply = score, reply
        features, key = secondary(root, move, recent)
        scored.append({"move": move, "primary_score": primary, "secondary": features,
                       "secondary_key": key,
                       "pv": [move] + ([worst_reply] if worst_reply else [])})
    ranked = sorted(scored, key=lambda row: (row["primary_score"], row["secondary_key"]),
                    reverse=True)
    best = ranked[0]
    tied = [row for row in scored if row["primary_score"] == best["primary_score"]]
    return dict(common, chosen_move=best["move"], primary_score=best["primary_score"],
                secondary=best["secondary"], pv=best["pv"], primary_ties=tied,
                top_candidates=ranked[:5], search_performed=True, completed_depth=2,
                candidates_evaluated=len(scored), simulator_calls=calls,
                reason="highest_primary_score" if len(tied) == 1 else "secondary_then_uci")
