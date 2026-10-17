# Search versus agent harness

Only `search` and `harness` are runnable comparison modes; `harness` is the default. Both use original r13
search with `development-king-safety-v2`, a 6,000-node ceiling and maximum depth 4.
The batch runner makes no outer LLM calls. `search` commits the search result;
`harness` lets DeepSeek plan analysis and choose an explicitly verified move.
New harness runs request `reasoning_effort="none"` to avoid exhausting completion
tokens before emitting action JSON. Config, trace and manifest record this setting;
older runs retain their original reasoning configuration. Search nodes, depth,
verification and the eight-call allowance are unchanged.

Requests use strict JSON Schema for the existing action protocol, including
allowed action names, field types and legal move identifiers. Optional commit
fields remain optional. OpenRouter requests require parameter support from the
selected endpoint; other compatible endpoints must support JSON Schema directly.
The native executor still validates every action, budget and commit. There is no
automatic action repair, JSON-mode fallback or code-selected move. Config and
manifest record the response format; each trace records its exact schema and
provider preferences. Historical runs keep their original transport metadata.

See the main [README](../../README.md) for the seven 10-game batches: two modes at
Stockfish Elo 1500/1800/2000, plus harness against the original minimax depth 2.
`--opponent stockfish` is the batch default; `--opponent minimax` reuses the original
opponent without installing Stockfish. Stockfish strength flags are rejected for
minimax. Each summary row identifies the opponent; config, state and manifest must
agree. Source snapshots, per-game trajectories, decisions and replay reports are
saved for both opponents. Minimax replies are replayed exactly.

The original report's results and legacy policy identifiers remain historical evidence;
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

## Long games and sandbox lifetime

The batch runner defaults to a 3,600-second sandbox lifetime per game. For slower
model providers or long games, pass `--sandbox-timeout 14400` to allow four hours
per game. This changes infrastructure lifetime, not search nodes, depth, or the
eight-call LLM allowance. Use the same lifetime for both comparison modes.
The configured lifetime is recorded in `experiment-config.json`.

If runtime shutdown loses its connection, confirmed sandbox termination is enough
to finish cleanup. Failure to terminate still raises an error. Replay failures
are saved as unvalidated reports and included in the batch summary, rather than
aborting the remaining games. A conservative benchmark loss for an incomplete or
unvalidated run is not a native chess loss; inspect completion and error fields.
