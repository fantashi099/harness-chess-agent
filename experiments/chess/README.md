# Fixed chess policy v2

This extension preserves the v2 skill's two-ply search, evaluation and tie-breaks.
A fixed Python helper runs inside the sandbox; the model requests each turn.
See the main [README](../../README.md) for setup, opponent options and run commands.

## Offline checks

```bash
uv run pytest -q experiments/chess/test_fixed_policy.py tests
uv run python scripts/check_fixed_chess_policy.py
```

The four regression positions must select `Re4`, `Qe5+`, `Qd4+` and `Rb1+`
over the recorded inferior moves under the same two-ply evaluation.
The checker writes `artifacts/assignment2/fixed-policy-regressions.json`.

## Replay a saved run

```bash
uv run python scripts/replay_fixed_chess_game.py \
  --result artifacts/assignment2/my-fixed-run/result.json
```

The validator uses a temporary chess app checkout with the Part 1 patch applied.
It checks legal history, every White move against the helper, the final FEN and
outcome. Minimax replies are reproduced at depth 2; Stockfish replies are checked
for legality. Replay uses no model calls or Modal sandboxes and requires the
helper and patch versions recorded in the manifest.

## Execution guarantees

- The model receives `select_and_play` and, when skills are loaded, `invoke_skill`.
- The harness supplies the observed position identity; the model cannot choose a
  FEN, override the move or call raw move, Python or simulation tools.
- The server holds the game lock throughout search and commit, including against
  browser moves and resets.
- Identity includes the game ID, FEN and full history. Stale requests are rejected;
  duplicate requests return the recorded decision without committing again.
- Search calls `simulate_position` directly inside the sandbox. `simulator_calls`
  counts simulations rather than HTTP requests.
- Observations contain summaries; full ties and top candidates are saved separately.

Each run saves `trajectory.json`, `result.json`, `decisions.jsonl`, `game.pgn`
and `manifest.json`. Results and decisions are saved in `finally`, including on
errors or step limits. Replay creates `replay-report.json`.

## Limitations and legacy mode

The helper does not implement deeper search, quiescence or full-history
repetition within search branches. Recent-move penalties only break ties.
Fixed execution does not guarantee a win or establish a general win rate.

The earlier model-written Python skill remains in
`legacy-skills/select-move-v2/SKILL.md`. Use
`--programmatic-tools --skills-path experiments/chess/legacy-skills` to run it.
The original assignment mode and skills under `tasks/` retain their interfaces.
