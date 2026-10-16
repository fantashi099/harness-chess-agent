"""Shared chess identity, material values and move tie-break features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

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
