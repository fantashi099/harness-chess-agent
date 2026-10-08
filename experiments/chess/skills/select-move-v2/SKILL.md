---
name: select-move-v2
description: Play White using the fixed v2 two-ply minimax helper. Invoke when select_and_play is available; the helper selects, verifies and commits one move, with auditable decision artifacts.
---

# Select move v2: fixed execution

This skill uses the fixed helper when running with `--fixed-chess-policy v2`.

1. Read the latest live state.
2. If `game_over` is true, stop.
3. Otherwise call `select_and_play` with `{}` once.
4. Read the returned decision summary, Black reply, and new state. Repeat.

The tool executes the checked Python helper inside the sandbox. It selects
the move using complete two-ply minimax, then commits that exact move while
holding the live game's lock. A request is bound to the observed game,
FEN and full history. A duplicate request does not commit twice.

The helper keeps the original v2 evaluation: terminal outcome, material,
centralization and check. Among exact primary-score ties it prefers castling,
minor-piece development and central pawn advances, then penalizes early major
piece moves, returning home and repeating/reversing recent White moves.
Only the standard initial board uses the fixed opening preference.

The model does not supply moves, FEN, search code or alternative heuristics.
Only `invoke_skill` (when skills are loaded) and `select_and_play` are exposed
in this mode. After a tool error, use the refreshed observation and retry the
same tool. Do not attempt direct `play_move`, `run_python` or hypothetical
search calls: they are unavailable in fixed mode.

Each observation includes a compact `chess_decision` summary. Complete
candidate ties and top candidates are retained in the per-game
`decisions.jsonl` artifact. `completed_depth=2` records complete two-ply
search; the opening preference is explicitly `completed_depth=0`.

This remains a two-ply policy. It does not guarantee a win and does not
implement full-history repetition search. Recent-move penalties are
tie-break heuristics, even though live request validation includes history.

For the earlier model-written Python experiment, use the preserved skill at
`experiments/chess/legacy-skills/select-move-v2/SKILL.md` with
`--programmatic-tools`. Offline hypothetical comparisons should call the
pure `select_move` helper with `simulate_position`; it never commits a move.
