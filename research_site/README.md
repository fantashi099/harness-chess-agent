# Chess research viewer

A local, static research site backed by sanitized historical artifacts. No backend,
model calls, Stockfish execution, account, build step, or new dependency is needed.
The existing Python environment supplies `python-chess`; browsers load ordinary ES modules.

## Local setup

From the repository root, install the existing project dependencies with `uv sync`.
Import the benchmark report, then serve **only this directory**:

```bash
uv run python -m assignment.viewer benchmark artifacts/assignment2/benchmark-report --default
uv run python -m http.server 8765 --bind 127.0.0.1 --directory research_site
```

Open [the main page](http://127.0.0.1:8765/) or
[the viewer directly](http://127.0.0.1:8765/viewer.html). The main page is the Vietnamese
technical blog, centered on the Stockfish 1500 evaluation; 2000 examples are in its
appendix. Blog and Trajectories share the dark palette and compact project navigation.
Game cards read the imported benchmark catalog; featured examples link to recorded
events (game 1 repair, game 9 mate, game 4 incomplete). The article is an editorial
snapshot of the V1 report, not a live scoreboard; re-check its figures when revising
the evaluation. Game cards and their outcomes always read the imported dataset. Blog navigation returns to
the originating section when session storage is available. Existing root URLs with
`#run=...&game=...&event=...` still redirect to the viewer without losing their hash.
Do not use `file://`: dataset fetching
requires HTTP. Bind to loopback for local inspection. Do not serve the repository root.
Generated `data/` is ignored by Git; keep private originals under `artifacts/`.

If an older browser tab shows an empty viewer with an `addEventListener` error,
refresh it once (Ctrl+Shift+R if needed). That indicates cached scripts referring to
the previous inspector tabs. Browser assets now have versioned URLs, and dataset
requests bypass the browser cache. Bump the matching HTML asset and JS module query
versions together whenever shipping changes to their shared interface.

The default dataset is `artifacts/assignment2/benchmark-report`: **LLM + Tools**
(the `harness` policy) first, then **Tools Only** (the `search` policy), each against
Stockfish at 1500 and 2000 Elo. New imports default to LLM + Tools at 1500 Elo.
The stored policy names and existing run/event URLs remain unchanged.
The dark viewer has Mode, Elo and Game selectors. The comparison strip shows White
wins/draws/losses at the selected Elo; unfinished games count as losses in this
summary. Stored completion flags and individual game labels retain their recorded values.
The header contains only Mode, Elo and Game selectors. Configuration and provenance
remain in the imported JSON files. Explicit historical event links still work.

The importer reads only the four immediate `stockfish-{1500,2000}/{search,harness}`
groups, excluding archived copies under `sources/`, reports and logs. It stages all
four groups before adding them to the catalog and never overwrites imported IDs.
Games are ordered numerically, including `game_10`. Originals are never modified.

| Elo | Tools Only | LLM + Tools |
| --- | --- | --- |
| 1500 | 4 wins / 1 draw / 5 losses | 6 wins / 2 draws / 2 losses |
| 2000 | 3 wins / 2 draws / 5 losses | 4 wins / 1 draw / 5 losses |

**The 2000 Harness set is retrospectively selected by outcome across three source
runs.** It is a case-study collection, not a representative win-rate estimate or a
controlled comparison. Its stored `selection.json` remains in the imported run manifest.

Pure Search stores candidate scores and budget at the decision root instead of
Harness feedback. The importer preserves those as search-result events before
commit/opponent events. The board shows total Search usage, or final verification
usage in Harness. Search's inspector shows recorded search evidence without
inventing prompts, thinking or model calls.

The earlier five-game v41 schema batch remains supported:

```bash
uv run python -m assignment.viewer import artifacts/assignment2/harness-v41-schema-20261016-214857 --run-id harness-v41-schema-20261016-214857 --default
```

`--default` stores the preferred run in the dataset catalog. Explicit run/game/event
links still take precedence. This complete batch has two White wins, two losses,
and a draw; all five games are recorded as complete. It is separate from the three
curated historical examples below and is not a controlled comparison with them.

All 670 provider responses in this batch omit thinking content. The 659 validated
actions have recorded rationales; the other 11 responses also retain their rationale
but were rejected before a validated action was stored. The trace contains actions, search feedback,
commit confirmations and errors. The ReAct view uses those stored fields and labels
rationale separately from thinking. No additional logging or game reruns are needed
for this view. Actual thinking can only be shown if the provider returns it and a
future run stores it; the existing results cannot recover it afterward.

The local demo was imported using these three actual historical games:

```bash
uv run python -m assignment.viewer import artifacts/assignment2/v6-r13-2-2000-VX7zbhTk --run-id original-r13 --games game_4
uv run python -m assignment.viewer import artifacts/assignment2/v6-safe2400-6r5YnAUJ --run-id early-harness-r2 --games game_1
uv run python -m assignment.viewer import artifacts/assignment2/v6-r13-pv-material-2000-86OupqTP --run-id pv-material-r13 --games game_2
```

These are curated examples: an r13/GLM 5.3 White win, an older r2/GLM 5.2 loss,
and a PV-material r13/GLM 5.3 incomplete game. Opponent settings differ (2000,
2400, 2000). This selection cannot measure comparative performance. The incomplete
game's historical summary called it a benchmark loss; this viewer deliberately uses
`result.completed` and terminal `replay_result` separately and leaves its outcome unavailable.

## Inventory and supported evidence

The repository contains many versioned batches under `artifacts/assignment2/`,
including original r13, r13 variants, and earlier v6 revisions. Inspection covered
r2, r7, original r13, and PV-material r13 records, plus the artifact-writing code.
The supplied ARC inspector informed the synchronized run/game/event workflow;
its React/Vinext stack is not required for this static dataset.

| Source | Recorded format | Viewer support |
| --- | --- | --- |
| `harness-trace.jsonl` | One model call per line; request messages, provider response, accepted/rejected action, feedback, duration | Split into context, assistant, request, feedback, verification, replan and rejection events; raw row preserved |
| `decisions.jsonl` | One White decision per line, root/post FEN, UCI move, opponent move, analysis trace, optional embedded harness records | Commit/opponent events; search-only or embedded-record fallback when separate harness trace is absent |
| `result.json` | Final state, completion flag, stop reason, move history, terminal result when available | Completion/outcome metadata; legal move-history fallback |
| `game.pgn` | Standard saved game | Legal replay fallback when detailed events and result history are absent |
| `manifest.json` | Harness/search/evaluator versions, configuration, model usage, source hashes | Provenance in the sanitized metadata file |
| `experiment-config.json`, `source-freeze.json` | Batch settings and source hash map | Version-independent run manifest; source-snapshot reference retained |
| `trajectory.json` | Outer driver steps or older transcript fields | Retained intact after sanitization in lazy metadata; no speculative outer-transcript parser |

Recorded prompts may contain prior rationale, feedback, and board facts. The viewer
does not invent memory or skills: inspect the request actually stored. Source snapshots
are referenced by names and hashes, never executed or automatically copied. Videos,
replay reports, arbitrary logs, and unrelated files are not ingested. Harness event
timestamps and dollar costs were absent from the inspected schemas. Assistant events
show the provider's `response.created` time where stored, explicitly labeled as
provider-created time; it does not timestamp tool execution. Per-call durations
are durations, not timestamps. Provider-returned `reasoning_content` is labeled as a stored
field and is never synthesized. A model's own `rationale` remains its recorded claim.

## Add or curate runs

```bash
uv run python -m assignment.viewer import artifacts/assignment2/ANOTHER_RUN --run-id stable-new-id --games game_1 game_3
```

Omit `--games` to import all `game_*` directories. A single game directory is also
accepted and exposed as `game_1`. `--output` selects a separate dataset directory;
the default is `research_site/data`. No frontend edits are needed. Existing IDs cannot
be overwritten. Use a new ID to revise an import; this preserves old event URLs.
The run and game selectors read the append-only catalog automatically.

The pipeline normalizes in memory, then writes a **separate sanitized dataset**:

```text
data/manifest.json                         versioned run catalog
data/runs/RUN/manifest.json                configuration + curated game summaries
data/runs/RUN/GAME/index.json              lightweight ordered event/board index
data/runs/RUN/GAME/search.json             retained search index (unused by the viewer)
data/runs/RUN/GAME/events/EVENT.json        selected readable evidence
data/runs/RUN/GAME/records/HASH.json        shared sanitized source rows
data/runs/RUN/GAME/metadata.json            sanitized manifest/result/trajectory + file hashes
```

## Normalized model and adapters

Schema version **1** is defined by `src/assignment/viewer/adapters.py`. An event
has a stable `id`, `type`, `title`, `turn`, source `file` and `location`, optional
`fen`, `board_phase`, `board_origin`, `move`, `candidates`, `budget`, and lazy `detail`
path. Null means unavailable, never zero. Event IDs derive from source location
and subevent type. Do not reorder/edit original artifacts if you want their IDs to persist.

`load_game()` is the adapter boundary. It supports harness JSONL (including earlier
optional fields), decision JSONL, then history/PGN. Add a new source adapter here
that returns the same game/index and payload mapping; leave sanitization in the
export layer. Preserve unfamiliar record fields in raw payloads. Add an explicitly
synthetic fixture and ordering/board tests before enabling another format.

Trace rows retain their order within a turn. Root FEN move numbers anchor turns
across trace and decision files; records without a valid anchor follow known turns
in their source order. A decision's `post_fen` includes the opponent reply. White's
commit position is reconstructed by legally applying the stored UCI move to its root
FEN; it is labeled accordingly. Candidates and PVs never update the actual board.
Illegal moves or missing roots yield unavailable positions. A post-FEN disagreement
produces a warning and retains the recorded post-FEN.

Missing PGNs do not matter when JSONL exists. Missing prompts/responses remain
missing; decisions can still support board/search inspection. Malformed JSONL lines
are skipped with location warnings (including a truncated final line). Invalid whole
JSON files fail the import before a run is written. Missing completion flags are
unavailable. Only `completed=true` with a recorded terminal result has an outcome.

## Viewer behavior

A compact toolbar contains the title and Mode, Elo and Game selectors.
Model, budget and outcome remain in the board and game selector.
Desktop has three independently scrolling panels with pointer and arrow-key resize
handles. Small screens use Timeline / Board / Trace tabs. Native controls and details
are keyboard accessible; Left/Right outside controls navigates turns. Previous/Next and First/Last also navigate turns. Play advances immediately, then one turn every 2
seconds; Pause, manual navigation or switching games stops playback. Playback stops
at the final turn, which selects the final recorded event and shows Win/Loss/Draw
on the board from the White agent’s perspective when a completed result is stored.
Incomplete games have no outcome badge. Analysis budget uses the turn’s last
verification feedback. The info tooltip beside Analysis budget explains shared
node visits, per-turn resets, reserves and depth; hover, focus or tap to open, and
Escape or clicking outside closes it. Candidate evidence uses its last search feedback with a
candidate table. These remain the same when inspecting individual events in that
turn. The committed White move is highlighted as Chosen. If it is outside the
recorded search shortlist, a separate chosen-move note appears without inventing
a score. Verification scores are not mixed into the search table. Missing feedback
stays unavailable. The board keeps the from/to squares of White’s and Black’s
last played moves highlighted in green, including during playback. Player strips show pieces captured in recorded play
up to the selected event, including en passant; promotions do not count as captures.
Up to four consecutive forward plies animate in order (including castling’s rook);
long jumps and rewinds fade, and reduced-motion preferences disable animations.
The board
has a text description for assistive technology. Copy event link preserves run/game/event:

```text
viewer.html#run=original-r13&game=game_4&event=decisions-jsonl-1-commit
```

Direct navigation and reload restore selection; unknown event IDs fall back visibly.
The timeline groups each game into turn rows, with recorded moves and event counts.
Selecting a turn shows its full ReAct sequence. Expand its event list to select an
individual event; expanded state is retained while navigating. The event-type filter
keeps matching turns and their matching events. Turn rows render immediately, while
event buttons are created only for expanded groups. Game indexes, event details and raw source rows
are fetched separately. Expandable trace blocks defer DOM formatting until opened.
The trace inspector has two keyboard-accessible tabs:

- **Reasoning & tool:** a compact list of color-coded accordion rows for the selected
  turn, initially collapsed: system prompt → user prompt → thinking / recorded rationale
  → tool call → result / error → status, then the next call. Identical consecutive
  request messages are reused; changed prompts appear at their actual position.
  Provider thinking appears only when stored; absent thinking rows are omitted,
  and rationale remains separate assistant output. Row sizes count UTF-8 bytes of the
  displayed content, not tokens. Expand a row for its content and source-call link.
  The selected call is marked at the left. Raw transcript at the bottom retains every
  complete source call, including repeated prompts and metadata. Recorded status is
  never inferred from feedback. Rejected actions include their recorded errors.
- **Event JSON:** the selected normalized event, its evidence, and its sanitized source
  call when available. The existing download control uses the same sanitized evidence.

Selecting a call in the inspector selects its timeline event and updates the board;
switching tabs retains event selection. Complete turn sequences include events before
and after the selected event; the board continues to represent that selected event.
Tool results are only the fields actually stored: a commit confirmation does not
pretend to be an archived full HTTP response. No artifact string is rendered as HTML.
No editorial annotations are mixed into recorded logs.
Older imported datasets remain compatible without reimporting.

## Review and prepare a public export

Nothing is uploaded, deployed, pushed, or published by these commands. Import is
not publication approval. Automated redaction covers sensitive field names, embedded
JSON, common credential formats, bearer tokens, URLs, email addresses, local absolute
paths, and private key blocks, including unknown fields and raw downloads.
All browser-facing JSON passes through the same recursive sanitizer. HTML/CSS/JS
contains no remote assets or analytics. Original artifacts are never rewritten.

Before publication, inspect **all** files in `data/`, especially raw rows, prompts,
search indexes and metadata, for sensitive prose and identifiers that pattern-based
redaction cannot recognize. Extend `sanitize.py` and its tests if needed, then regenerate
into a separate output directory for review. Do not assume a known-pattern scanner can
identify every possible secret. Only after human review, prepare a local package:

```bash
uv run python -m assignment.viewer export --output /tmp/chess-reviewed-export --reviewed
```

The destination must not exist. Only known browser assets and sanitized `data/`
are copied; source files, README, tests, `.env`, and original artifacts are excluded.
A SHA-256 inventory is written **beside** the package to identify reviewed bytes.
This flag confirms review for local packaging; deployment still requires your approval.

## Checks and limits

```bash
uv run python -m pytest -q tests/viewer
node research_site/model.test.mjs
node research_site/blog.test.mjs
node --check research_site/app.js
```

Python tests cover adapters/older fields, ordering, missing/partial records, unknown
fields, source immutability, board transitions, redaction, curated outcomes, adding
a second run, and export allowlisting. Node checks cover deep links, missing events,
and FEN-to-square synchronization, configured default selection, call deduplication,
transcript ordering, rejected actions, and absent versus recorded thinking. Node 22+
is only needed for these optional checks.
No real model, engine, or Modal test is needed to view/import old evidence.

Limits: opening every turn's event list
can eventually produce a large DOM. New transcript-only formats need an adapter.
There is no interactive PV replay, cross-run causal analysis, or inferred Elo/cost.
The next useful increment is another tested adapter if new artifacts require it, or
a full-batch comparison with matched model/opponent/budget settings.
