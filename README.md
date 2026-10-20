# Chess Agent Harness: CMU Assignment Extension

> Shout-out to **Weiwei Sun, Saujas Vaduguru, and the Carnegie Mellon University
> 11-768 course staff** for creating the original agent-harness assignment and
> chess project that made this work possible. Full acknowledgments are below.

This project extends CMU's Assignment 1 to improve a chess-playing agent and
evaluate it against **Stockfish** at different Elo settings, alongside the
original minimax opponent. It explores how an LLM can use bounded search,
tactical verification, and structured feedback to choose better moves.
Improvement is the experimental goal; a higher win rate must be established
through benchmarks.

The original coding-agent workflow is also retained: a shared ReAct agent loop
supports terminal tools, context compaction, and repairing the chess app before
playing it. Original requirements and submission instructions are preserved in
[ASSIGNMENT.md](ASSIGNMENT.md). The chess extensions are personal experiments,
not official CMU course requirements.

## How the chess harness works

The harness combines an LLM planner with Python search and an executor that
checks actions before applying them. White is the agent; Black is Stockfish or
the original minimax bot.

1. **Observe:** the LLM receives the current board, legal moves, recent history,
   remaining budget, and feedback from previous analyses at this position.
2. **Analyze:** it requests a broad search over legal moves, then focuses on
   individual candidates. The search uses iterative-deepening alpha-beta,
   quiescence, and a positional evaluator covering material, development,
   king safety, mobility, and pawn structure.
3. **Verify:** single-candidate analysis checks Black's replies and performs
   bounded mate probes. Feedback includes completed depth, score bounds,
   principal variations, and tactical refutations. These checks do not prove
   general safety.
4. **Commit:** the LLM explicitly chooses an analyzed move through structured
   JSON. The executor checks legality, verification requirements, and budgets;
   incomplete verification or known refutations require eligible explicit risk
   acceptance. It does not silently substitute a move.
5. **Record:** the runner saves decisions, model traces, source snapshots, game
   results, and replay reports so each experiment can be inspected offline.

By default, each White turn shares a **6,000-node budget** across search and
safety probes, with a maximum main-search depth of **4 plies**, at most four
analyses, and eight LLM calls including rejected actions. Nodes are not a fresh
allowance for every tool call. Quiescence can extend beyond the main-search depth.

| Component | Responsibility |
| --- | --- |
| `src/assignment/agent/chess_agent.py` | Connects the agent to chess tools |
| `src/assignment/agent/chess_harness.py` | Builds model requests and action schemas |
| `src/assignment/agent/chess_policy_harness.py` | Enforces the analysis and commit protocol |
| `src/assignment/agent/chess_policy_search.py` | Implements search and position evaluation |
| `experiments/chess/evaluate_search_policy.py` | Orchestrates benchmark batches in Modal |

## Setup

Install Python 3.11 or newer and [uv](https://docs.astral.sh/uv/), then run:

```bash
make setup
uv run modal setup
cp .env.example .env
```

Configure the model endpoint, API key and model in `.env`. Keep credentials out
of Git. Run `make doctor` to check source revisions and service configuration.
Modal sandboxes and model calls consume credits; offline tests do not.

## Run benchmarks

Choose how White selects its moves:

| `--policy` | Final move authority | Model calls |
| --- | --- | --- |
| `harness` | LLM requests analysis, reads feedback, and explicitly commits a move | Yes |
| `search` | Search picks the best completed result | None |

Both modes use the same search implementation and positional evaluator. The
examples use a 6,000-node ceiling and maximum main-search depth of 4 plies.
`harness` is the default policy. Use a fresh output directory for each batch.

### Harness against Stockfish

After setup, run a harness batch against Stockfish Elo 2000:

```bash
BATCH="artifacts/assignment2/benchmark-$(date +%Y%m%d-%H%M%S)"
MODEL="deepseek/deepseek-v4.1-flash"

uv run python -m experiments.chess.evaluate_search_policy \
  --policy harness --opponent stockfish --stockfish-elo 2000 \
  --model "$MODEL" --games 5 --steps 200 \
  --max-nodes 6000 --depth 4 --sandbox-timeout 14400 \
  --output "$BATCH/stockfish-2000/harness"
```

### Harness against minimax

In the same shell, run the harness against the original minimax depth-2 opponent:

```bash
uv run python -m experiments.chess.evaluate_search_policy \
  --policy harness --opponent minimax \
  --model "$MODEL" --games 5 --steps 200 \
  --max-nodes 6000 --depth 4 --sandbox-timeout 14400 \
  --output "$BATCH/minimax/harness"
```

The examples select the model explicitly; change `MODEL` for another supported
model. `--depth` controls White's search,
not the minimax opponent's depth. Omit Stockfish strength flags for minimax.
`--sandbox-timeout` is the lifetime per game, not the timeout for the whole batch.

### Compare harness with search

For a search-only control, use `--policy search` with the same opponent, node
budget, and depth, and choose another output directory. Budget allocation and
verification are part of the harness treatment; report actual node use and model
cost alongside game outcomes.

Run the full comparison across three Stockfish Elo settings and minimax:

```bash
BATCH="artifacts/assignment2/ablation-$(date +%Y%m%d-%H%M%S)"
for ELO in 1500 1800 2000; do
  for POLICY in search harness; do
    uv run python -m experiments.chess.evaluate_search_policy \
      --policy "$POLICY" --opponent stockfish --output "$BATCH/$ELO/$POLICY" \
      --model deepseek/deepseek-v4.1-flash \
      --games 10 --max-nodes 6000 --depth 4 --stockfish-elo "$ELO" \
      --sandbox-timeout 14400
  done
done

uv run python -m experiments.chess.evaluate_search_policy \
  --policy harness --opponent minimax --output "$BATCH/minimax/harness" \
  --model deepseek/deepseek-v4.1-flash \
  --games 10 --max-nodes 6000 --depth 4 --sandbox-timeout 14400
```

This schedules 60 Stockfish games and 10 harness games against the original
Assignment 1 minimax depth-2 opponent. The batch runner defaults to 10 games and
Stockfish at Elo 1500; select minimax explicitly and omit Stockfish strength flags.
Both modes play White from the same initial board. Stockfish runs record the same
opponent settings: Skill Level 20 with Elo limiting at the requested target.
`UCI_Elo` takes precedence over Skill Level when strength limiting is enabled;
these are not unlimited-strength games.
Timed Stockfish replies can vary. These runs consume Modal credits; harness mode
also consumes model credits. Use a fresh output directory for each batch.
The ordinary agent CLI exposes the same choices as `--chess-policy search|harness`.
`harness` is the default in both commands. Use `--policy search` for the batch
control, or `--chess-policy search` in the agent CLI. `--programmatic-tools`
selects the original Assignment 1 Python tool mode.

The supplied `.env.example` sets `OPENAI_MODEL` to
`deepseek/deepseek-v4-flash-0731`; the benchmark commands above override it.
See the [experiment guide](experiments/chess/README.md) for artifact details and
offline validation. Recorded source snapshots and model metadata identify the
implementation used for each run.

## Other ways to play

### Original assignment workflow

Generate and check the Part 1 patch, then play against the default minimax opponent:

```bash
make run-code-agent
make check-part1
make run-chess-agent
```

See [ASSIGNMENT.md](ASSIGNMENT.md) for context compaction, SWE-bench,
programmatic chess tools and observation experiments.

### Stockfish in Modal

Minimax remains the default. Add `--opponent stockfish` to either Modal runner:

```bash
uv run assignment-chess-modal --opponent stockfish --patch artifacts/fix.patch
uv run assignment-play-chess --opponent stockfish --patch artifacts/fix.patch \
  --trajectory artifacts/stockfish/trajectory.json --result artifacts/stockfish/result.json
```

These commands run the board, server and Stockfish **inside Modal**, using CPU
only. The optional image layer installs the distribution's Stockfish package;
the actual engine name/version is recorded in game state. Your laptop runs the
runner and displays the browser. Agent runs also use model API credits.

### Local browser game

Install Stockfish and the chess app dependencies:

```bash
sudo apt-get install stockfish
uv pip install -e chess_app
PYTHONPATH=chess_app/src uv run python -m assignment.chess_server --opponent stockfish
```

Open http://127.0.0.1:8000. Local Stockfish mode handles terminal moves without
the Part 1 patch. The existing `chess-server` command keeps its original behavior;
use the extension command above to select Stockfish.

### Opponent settings

The Modal and local server commands support `--stockfish-time 0.2` (default;
maximum 10 seconds) and
`--stockfish-elo 1500` (optional; accepted range depends on the installed engine).
Stockfish uses one thread and a 64 MiB hash table. `--stockfish-path /path/to/stockfish`
selects an executable **on the server host**; a laptop path is unavailable in Modal.
For a downloaded local version, pass its executable through this flag.

Stockfish mode also works with `--chess-policy search` and `--chess-policy harness`. Saved results include
engine settings; fixed-run manifests and PGNs identify Stockfish. Its time-limited
search and Elo limiting can produce different replies between runs, so the fixed
replay checker validates Black's legal moves rather than reproducing them.
Minimax runs retain exact opponent replay. Simulation remains a single legal ply
with no automatic opponent reply.

Free checks, without launching Modal or calling a model:

```bash
uv run python scripts/check_stockfish_opponent.py
uv run python scripts/check_stockfish_opponent.py --stockfish-path /usr/games/stockfish
```

## Offline validation and replay

```bash
make test
uv run python -m pytest -q tests experiments/chess
uv run python scripts/replay_chess_game.py \
  --result artifacts/assignment2/RUN/game_1/result.json --report /tmp/replay-report.json
```

Replay requires the original helper version, Part 1 patch and manifest.
It reproduces minimax replies; timed Stockfish replies are checked for legality.
Generated trajectories, reports and recordings belong under `artifacts/`.

To render a saved game, install `ffmpeg` and run:

```bash
uv run python scripts/record_chess_game.py \
  --result artifacts/assignment2/RUN/game_1/result.json \
  --out artifacts/assignment2/RUN/game_1/game.mp4
```

## Repository layout

The [local chess research viewer](research_site/README.md) imports historical runs
without model calls or replay execution. It provides synchronized harness events,
board states, sanitized trace inspection, and stable event links. See its guide for
offline imports, local preview, and the required review before a public export.

| Path | Purpose |
| --- | --- |
| `src/assignment/agent/` | Shared agent loop, tools and chess policy |
| `src/assignment/` | CLI, sandbox runners, servers and artifact handling |
| `scripts/` | Validation, replay and recording commands |
| `tests/` | Assignment regression tests |
| `experiments/chess/` | Controlled evaluation runner and chess tests |
| `tasks/` | Original task definitions and skills |
| `chess_app/` | Pinned upstream app submodule |
| `artifacts/` | Local run outputs and evidence |

## Credits

The original CMU assignment and chess project were created by **Weiwei Sun and
Saujas Vaduguru**, with feedback from the 11-768 course staff:

- **Instructors:** Daniel Fried and Graham Neubig.
- **Teaching assistants:** Aditya Soni, Andy Liu, Apurva Gandhi, Demi Wang,
  Jiarui Liu, and Yueqi Song.
