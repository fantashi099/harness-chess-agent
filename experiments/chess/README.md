# Chess policy experiments

The final implementation uses original v6-r13: policy
`select-move-v6-llm-harness-13`, search `iterative-alpha-beta-q6-2` and evaluator
`development-king-safety-v2`. The default budget is 6,000 nodes per turn at depth 4.
The LLM chooses the final move; search supplies bounded evidence.

## Run original r13

Configure `OPENAI_MODEL=deepseek/deepseek-v4-flash-0731` in your local `.env`.

```bash
uv run python -m experiments.chess.evaluate_search_policy \
  --policy v6 --output artifacts/assignment2/r13-deepseek \
  --games 5 --max-nodes 6000 --depth 4 --stockfish-elo 2000
```

This starts billable Modal sandboxes and real model calls. Use a fresh output
directory. Every requested game is recorded, including failures and incomplete games.
The original report batch used GLM 5.3; its results do not measure DeepSeek performance.
Archived results and source snapshots retain their original model metadata.

## Offline tests and replay

```bash
uv run python -m pytest -q tests experiments/chess
uv run python scripts/check_stockfish_opponent.py
uv run python scripts/check_fixed_chess_policy.py --output /tmp/fixed-policy-regressions.json
uv run python scripts/replay_v3_chess_game.py \
  --result artifacts/assignment2/RUN/game_1/result.json --report /tmp/replay-report.json
```

Replay verifies source hashes, legal history, search evidence and explicit LLM
commits without calling a model. Stockfish replies are checked for legality and
recorded configuration; timed replies are not reproduced exactly.

The v2 baseline and legacy skill remain available. v3/v4 modules are retained for
existing CLI and replay dependencies. Later revisions, recovery/order/cache patches,
endgame skills and development notes are kept separately from this active codebase.
See the main [README](../../README.md) for setup, recording and the original assignment.
