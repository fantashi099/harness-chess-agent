# Chess trajectories

Standalone, static explorer for **20 recorded games** from the LLM chess agent,
against Stockfish at 1500 and 2000 Elo, with 10 games at each Elo.
The root page contains Elo and Game selectors, timeline, board and trace inspector.
There is no mode selector or results comparison strip.
There is no blog, backend, model call or engine execution on this website.

Public site: https://fantashi099.github.io/chess-trajectories/

Published repository: https://github.com/fantashi099/chess-trajectories

The unfinished blog and its related code are preserved in branch
`tmp/chess-blog-snapshot`, commit `0264778`. Continue it there when ready.
The standalone website lives on `feat/chess-trajectories`.

## Run locally

```bash
uv sync
uv run python -m assignment.viewer benchmark artifacts/assignment2/benchmark-report --default
uv run python -m http.server 8765 --bind 127.0.0.1 --directory research_site
```

Import once into a fresh `research_site/data` directory. Existing IDs are never
overwritten. Open http://127.0.0.1:8765/ over HTTP. Serve only this directory,
never the repository root. Legacy `viewer.html#run=...&game=...&event=...` links
redirect to the root while preserving their selection.

Generated data stays ignored by Git. Earlier imported demos are retained locally
under `.private/chess-viewer-extra-runs`, outside the public site. Original run
artifacts are never modified. Rebuilding from the benchmark imports only the four
immediate policy/Elo groups; nested source archives are excluded. The viewer only
loads Harness runs; the public catalog and publication contain only the two Harness groups.

## Recorded collection

| Elo | Recorded games |
| --- | --- |
| 1500 | 10 |
| 2000 | 10 |

Individual game labels retain completion status. **The 2000 Elo group was selected
retrospectively by outcome across three source runs.** It is a case-study collection,
not a representative win-rate estimate. Selection provenance stays in its run manifest.
These recordings do not demonstrate Reflexion or learning between games.

Trace fields are recorded evidence. Missing thinking is never synthesized;
a model's rationale remains its claim. Board positions and playback follow actual moves,
not hypothetical principal variations. Per-call durations are not timestamps.

## Export and publish

```bash
uv run python -m assignment.viewer export --output .private/chess-trajectories-export --reviewed
```

Review the sanitized dataset before export. The export allowlist contains only
`index.html`, `viewer.html`, `styles.css`, `app.js`, `model.js` and `data/`.
Unused generated `search.json` indexes are excluded; the viewer reads event indexes
and source records directly. All recorded events, prompts and reasoning remain.
No blog, environment file, original artifact, source snapshot or server code is
published. The importer redacts secrets, credentials, personal identifiers, URLs
and local filesystem paths, including JSON embedded in transcripts.

The GitHub Pages publishing checkout is `.private/github-chess-trajectories`,
a separate repository containing only public browser assets and sanitized data.
Pages serves the root of its `main` branch; `.nojekyll` disables Jekyll. All URLs
are relative so the viewer works under `/chess-trajectories/`. Copy a reviewed
export into that checkout, retaining `.git`, `.nojekyll` and its public README,
then commit and push to publish updates. Choose a fresh export destination.

The previous Sites checkout remains at `.private/chess-trajectories` as a backup.

## Checks

```bash
node --test research_site/model.test.mjs
uv run pytest tests/viewer -q
```

The importer preserves stable event IDs and lazy per-event payloads. Schema
version 1 and record adapters live in `src/assignment/viewer/adapters.py`;
redaction lives in `src/assignment/viewer/sanitize.py`.
