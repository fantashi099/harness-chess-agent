# Search versus agent harness

Only `search` and `harness` are runnable comparison modes; `harness` is the default. Both use original r13
search with `development-king-safety-v2`, a 6,000-node ceiling and maximum depth 4.
The batch runner makes no outer LLM calls. `search` commits the search result;
`harness` lets DeepSeek plan analysis and choose an explicitly verified move.

See the main [README](../../README.md) for matched batch commands. The original
report's GLM results and legacy policy identifiers remain historical evidence;
old v5 scores used a different evaluator and cannot serve as this matched control.

## Offline checks

```bash
make test
uv run python scripts/check_stockfish_opponent.py
uv run python scripts/replay_chess_game.py \
  --result artifacts/assignment2/RUN/game_1/result.json --report /tmp/replay-report.json
```

Replay checks source hashes, legal history, search evidence and explicit model
commits. Stockfish replies are checked for legality and recorded configuration,
not reproduced exactly. Archived search/harness implementations are read without
altering snapshots. Obsolete executable policies, memory planners and their skills
are kept outside the active codebase; the original assignment remains unchanged.

Report all scheduled games, failures, incomplete games, node use and model cost.
Matched search settings isolate the contribution of harness orchestration and
verification; outcomes still require repeated full-game measurements.
