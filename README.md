# Agent Harness: Coding and Chess

An implementation of Carnegie Mellon University's 11-768 Assignment 1, with
optional chess policy and Stockfish extensions. Original requirements and
submission instructions are preserved in [ASSIGNMENT.md](ASSIGNMENT.md).
The extensions are personal experiments, not official course requirements.

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

## Run agents

Generate and check the Part 1 patch, then play against the default minimax opponent:

```bash
make run-code-agent
make check-part1
make run-chess-agent
```

See [ASSIGNMENT.md](ASSIGNMENT.md) for context compaction, SWE-bench,
programmatic chess tools and observation experiments.

## Fixed chess policy

The optional v2 policy runs a deterministic two-ply helper inside the sandbox.
The model requests turns through `select_and_play`; the helper selects and
commits each move.

```bash
uv run assignment-play-chess \
  --model glm-5.2 --fixed-chess-policy v2 \
  --skills-path experiments/chess/skills \
  --patch artifacts/fix.patch \
  --trajectory artifacts/assignment2/my-fixed-run/trajectory.json \
  --result artifacts/assignment2/my-fixed-run/result.json
```

Use a fresh output directory for each run. Do not combine `--fixed-chess-policy`
with `--programmatic-tools`. See the [experiment guide](experiments/chess/README.md)
for execution guarantees and limitations.

## Optional Stockfish opponent

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

For a **local** browser game, install Stockfish and the chess app dependencies:

```bash
sudo apt-get install stockfish
uv pip install -e chess_app
PYTHONPATH=chess_app/src uv run python -m assignment.chess_server --opponent stockfish
```

Open http://127.0.0.1:8000. Local Stockfish mode handles terminal moves without
the Part 1 patch. The existing `chess-server` command keeps its original behavior;
use the extension command above to select Stockfish.

Both environments support `--stockfish-time 0.2` (default; maximum 10 seconds) and
`--stockfish-elo 1500` (optional; accepted range depends on the installed engine).
Stockfish uses one thread and a 64 MiB hash table. `--stockfish-path /path/to/stockfish`
selects an executable **on the server host**; a laptop path is unavailable in Modal.
For a downloaded local version, pass its executable through this flag.

Stockfish mode also works with `--fixed-chess-policy v2`. Saved results include
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
uv run pytest -q experiments/chess/test_fixed_policy.py
uv run python scripts/check_fixed_chess_policy.py
uv run python scripts/replay_fixed_chess_game.py \
  --result artifacts/assignment2/my-fixed-run/result.json
```

Replay requires the original helper version, Part 1 patch and manifest.
It reproduces minimax replies; timed Stockfish replies are checked for legality.
Generated trajectories, reports and recordings belong under `artifacts/`.

To render a saved game, install `ffmpeg` and run:

```bash
uv run python scripts/record_chess_game.py \
  --result artifacts/assignment2/my-fixed-run/result.json \
  --out artifacts/assignment2/my-fixed-run/game.mp4
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/assignment/agent/` | Shared agent loop, tools and chess policy |
| `src/assignment/` | CLI, sandbox runners, servers and artifact handling |
| `scripts/` | Validation, replay and recording commands |
| `tests/` | Assignment regression tests |
| `experiments/chess/` | Extension skills, fixtures and tests |
| `tasks/` | Original task definitions and skills |
| `chess_app/` | Pinned upstream app submodule |
| `artifacts/` | Local run outputs and evidence |

## Credits

Authors: Weiwei Sun and Saujas Vaduguru, with feedback from 11-768 course staff (instructors Daniel Fried and Graham Neubig, and TAs Aditya Soni, Andy Liu, Apurva Gandhi, Demi Wang, Jiarui Liu, and Yueqi Song).
