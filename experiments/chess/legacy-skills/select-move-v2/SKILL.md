---
name: select-move-v2
description: Choose a White chess move with two-ply minimax, explicit development and castling tie-breaks, and an auditable decision log. Use with run_python and simulate_move; supports simulation-only comparisons.
---

# Select move v2

Use this skill when playing White with programmatic chess tools, or comparing
move selection on a supplied FEN. The first improvement over v1 is deterministic
selection among equally scored moves, with evidence in stdout.

## Tool contract

- `simulate_move(fen, move=None)` inspects a position or applies exactly one
  legal ply. It returns `fen`, `squares`, `turn`, `legal_moves`, `in_check`,
  `game_over`, `winner` and `result`. It is stateless, works for either side,
  never invokes Black's bot, and never changes the live game.
- `play_move(move)` commits a White move on the live board; Black replies
  automatically unless White ended the game.
- Both functions are already available inside `run_python`. Use their returned
  fields; do not write a board-layout FEN parser or implement attack detection.

Use the newest observed live FEN for a live turn. Set `commit = False` when the
user requests analysis, comparison, or a supplied hypothetical position.
In a live game, use `commit = True` and call `play_move` exactly once, as the
last action of the snippet. An analysis result is not a move on the real board.

## Inputs for each snippet

Set these inputs explicitly before using the reference implementation:

```python
fen = "<latest observed FEN, or the FEN being analyzed>"
recent_white_moves = []  # Up to four confirmed White UCI moves, oldest first.
recent_moves_available = False
commit = True           # False for a simulation-only comparison.
```

Populate `recent_white_moves` only from confirmed successful White moves in the
available transcript, for example `human_move` in post-move observations.
Failed calls and simulated moves do not count. If that evidence is unavailable,
leave the list empty and `recent_moves_available = False`. At a confirmed new
game, an empty list with `recent_moves_available = True` is appropriate.
Do not invent UCI history from a shortened SAN history.

## 1. Preserve the opening baseline

Only at the standard initial position, use the first legal move in
`e2e4`, `d2d4`, `c2c4`, `g1f3`. Log that this is an opening preference and
`search_performed = False`; do not invent candidate scores.
Any other position, including a custom FEN with fullmove number 1, uses search.

## 2. Search exactly two plies

Inspect the root and enumerate all legal White candidates, sorted by UCI.
For each candidate, simulate White's move and every legal Black reply. The
candidate's primary score is the minimum leaf evaluation across Black replies.
White selects the maximum of those candidate scores.

Stop at terminal states and evaluate them directly. A nonterminal node with no
legal replies is an inconsistent observation: report the error, do not silently
assign a score or commit from incomplete search. If a tool fails, let the snippet
report its error and recover on the next agent turn.

The score is unchanged from v1, from White's perspective:

| Component | Score |
|---|---|
| Terminal White win | +100,000 |
| Terminal Black win | -100,000 |
| Terminal draw reported by simulator | 0 |
| Material | White minus Black: pawn 100, knight 320, bishop 330, rook 500, queen 900, king 0 |
| Centralization | +12 per White piece and -12 per Black piece on d4, e4, d5, e5 |
| Check | +25 when Black is in check, -25 when White is in check |

Keep the Black reply achieving the minimum as a two-ply principal variation.
When several replies tie, keep the first UCI reply. This is a worst-case reply
under this evaluation, not a prediction of the actual bot's chosen move.

## 3. Break exact primary-score ties

Only compare secondary features when primary scores are exactly equal.
Calculate features from the root board and candidate UCI, with legal moves
already validated by the simulator.

Maximize this tuple lexicographically, in this exact order:

```text
(castle, develop_minor, advance_center,
 -early_major_penalty, -return_home_penalty, -recent_move_penalty)
```

- `castle`: 1 for a White king's legal `e1g1` or `e1c1`, otherwise 0.
- `develop_minor`: 1 for a knight leaving b1/g1 or bishop leaving c1/f1.
- `advance_center`: 1 for `d2d3`, `d2d4`, `e2e3` or `e2e4` by a White pawn.
- `early_major_penalty`: 1 for a non-castling king, queen or rook move during
  fullmove numbers 1–10. Reading the FEN move counter does not require parsing
  its board layout.
- `return_home_penalty`: 1 for a knight, bishop, rook or queen moving from
  outside that piece type's starting squares back to one of them.
- `recent_move_penalty`: 1 if the UCI repeats any recent White move, plus 1 if it
  reverses any recent White move. This is a move-cycle heuristic, not a chess
  repetition detector.

If the complete secondary tuples also tie, keep the first UCI candidate.
These priorities match the programmatic system prompt. Never add the tuple
to the primary score or override a higher primary score for a strategic preference.

## 4. Print the decision, then commit once

Print one `chess_decision` record containing:

- Root FEN, whether search ran, and whether recent-move evidence was available.
- Chosen move, primary score, secondary feature values, and principal variation.
- Every candidate tied for best primary score with its features.
- Up to five highest-ranked candidates with scores and principal variations.
- Simulator-call count, selection reason, and whether the snippet will commit.

The record is a Python dictionary printed to stdout, not guaranteed JSON.
It is retained in the existing `run_python` tool observation. No new logging
tool or runner is required.

## Reference implementation

Use the following code after setting the inputs above. Keep the same evaluation,
tie-break order and log fields when adapting it; do not replace the tie-break
with an informal judgment. It uses built-in Python and the two injected tools.

```python
VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}
CENTERS = {"d4", "e4", "d5", "e5"}
HOMES = {"N": {"b1", "g1"}, "B": {"c1", "f1"},
         "R": {"a1", "h1"}, "Q": {"d1"}}
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEATURE_NAMES = ("castle", "develop_minor", "advance_center",
                 "early_major_penalty", "return_home_penalty", "recent_move_penalty")
simulator_calls = 0

def simulate(fen, move=None):
    global simulator_calls
    simulator_calls += 1
    return simulate_move(fen, move)

def evaluate(state):
    if state["game_over"]:
        winner = state["winner"]
        return 100000 if winner == "white" else -100000 if winner == "black" else 0
    score = 0
    for square, piece in state["squares"].items():
        if piece == ".":
            continue
        value = VALUES[piece.upper()] + (12 if square in CENTERS else 0)
        score += value if piece.isupper() else -value
    if state["in_check"]:
        score += 25 if state["turn"] == "black" else -25
    return score

def secondary(root, move, recent):
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

def choose_move(fen, recent):
    root = simulate(fen)
    if root["game_over"]:
        print("chess_decision", {"root_fen": fen, "chosen_move": None,
              "search_performed": False, "reason": "terminal_position", "commit": False})
        return None
    if root["turn"] != "white":
        raise ValueError("This skill selects moves only for White.")
    candidates = sorted(root["legal_moves"])
    if not candidates:
        raise ValueError("Nonterminal White root has no legal moves.")
    common = {"skill": "select-move-v2", "root_fen": root["fen"],
              "recent_white_moves": recent,
              "recent_moves_available": recent_moves_available,
              "repetition_history_available": False, "commit": commit}
    if root["fen"] == INITIAL_FEN:
        best = next(move for move in ("e2e4", "d2d4", "c2c4", "g1f3")
                    if move in candidates)
        features, _ = secondary(root, best, recent)
        print("chess_decision", dict(common, chosen_move=best, primary_score=None,
              secondary=features, pv=[best], primary_ties=[], top_candidates=[],
              search_performed=False, simulator_calls=simulator_calls,
              reason="opening_preference"))
        return best
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
        scored.append({"move": move, "primary_score": primary,
                       "secondary": features, "secondary_key": key,
                       "pv": [move] + ([worst_reply] if worst_reply else [])})
    # Stable sort retains ascending UCI order for fully equal ranking keys.
    ranked = sorted(scored, key=lambda row: (row["primary_score"], row["secondary_key"]),
                    reverse=True)
    best = ranked[0]
    tied = [row for row in scored if row["primary_score"] == best["primary_score"]]
    print("chess_decision", dict(common, chosen_move=best["move"],
          primary_score=best["primary_score"], secondary=best["secondary"], pv=best["pv"],
          primary_ties=tied, top_candidates=ranked[:5], search_performed=True,
          completed_depth=2, simulator_calls=simulator_calls,
          reason="highest_primary_score" if len(tied) == 1 else "secondary_then_uci"))
    return best["move"]

best = choose_move(fen, recent_white_moves[-4:])
if best is not None and commit:
    play_move(best)
```

After a live commit, use the refreshed observation for the next turn. Never
submit the chosen move a second time because it also appeared in stdout.

## Limits and next experiment

This version still searches two plies. It does not implement deeper search,
full-history repetition detection, memory persistence, or bounded search.
Recent-move penalties only influence equally scored candidates and may miss a
cycle. A FEN-only simulator cannot detect threefold repetition from past play.
A terminal draw reported by the simulator is scored 0, but the live server may
detect additional history-dependent draws.

For the first comparison, run v1 and v2 on the same nonterminal White FEN with
`commit = False`. Compare `chosen_move`, `primary_score` and `primary_ties`.
When the best primary score is unique, expect both versions to select the same
move. A different move is justified only by equal primary scores and the
explicit secondary ranking. A full-game win remains an outcome to measure.
