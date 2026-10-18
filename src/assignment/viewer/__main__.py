"""Offline import and reviewed static-export commands."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from .adapters import SCHEMA_VERSION, load_game, read_json
from .sanitize import sanitize


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(sanitize(value), ensure_ascii=False, separators=(",", ":")) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def import_run(source, output, run_id=None, games=None, make_default=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError("Artifact directory does not exist")
    run_id = run_id or source.name
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Run ID must contain only letters, numbers, underscores and hyphens")
    directories = sorted((p for p in source.glob("game_*") if p.is_dir()),
                         key=lambda p: (int(p.name[5:]) if p.name[5:].isdigit() else float("inf"), p.name))
    if not directories:
        directories = [source]
    if games:
        directories = [p for p in directories if p.name in games]
        if set(games) != {p.name for p in directories}:
            raise ValueError("Selected game directory is missing")
    if not any(any((p / f).is_file() for f in ("decisions.jsonl", "harness-trace.jsonl", "result.json", "game.pgn")) for p in directories):
        raise ValueError("No supported chess artifacts found")
    target = output / "runs" / run_id
    if target.exists():
        raise ValueError("Run ID already imported; use a new ID to preserve existing links")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and export directories must be separate")
    catalog_path = output / "manifest.json"
    catalog = read_json(catalog_path, dict(schema_version=SCHEMA_VERSION, runs=[]))
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported catalog schema version")
    if any(r["id"] == run_id for r in catalog["runs"]):
        raise ValueError("Run ID already imported; use a new ID to preserve existing links")
    config = read_json(source / "experiment-config.json", {})
    freeze = read_json(source / "source-freeze.json", {})
    run = dict(id=run_id, title=run_id, schema_version=SCHEMA_VERSION,
               source_directory=source.name, configuration=config, source_snapshot=freeze, games=[])
    # Parse every selected game before writing anything, so invalid input leaves no partial run.
    parsed = [(directory, *load_game(directory)) for directory in directories]
    try:
        for directory, game, payloads in parsed:
            game_id = directory.name if directory != source else "game_1"
            if not re.fullmatch(r"[A-Za-z0-9_-]+", game_id):
                raise ValueError("Unsafe game directory name")
            game_dir = target / game_id
            search = {}
            raw_records = {}
            for event in game["events"]:
                search[event["id"]] = event.pop("search")
                payload = payloads[event["id"]]
                record = payload["evidence"].pop("record", None)
                if record is not None:
                    raw_id = hashlib.sha256((event["source"]["file"] + event["source"]["location"]).encode()).hexdigest()[:20]
                    payload["raw_record"] = f"records/{raw_id}.json"
                    raw_records[raw_id] = record
                write_json(game_dir / event["detail"], payload)
            for raw_id, record in raw_records.items():
                write_json(game_dir / "records" / f"{raw_id}.json", record)
            write_json(game_dir / "search.json", search)
            # Metadata stays available, including trajectory fields and unknown fields, in a lazy file.
            write_json(game_dir / "metadata.json", dict(manifest=game.pop("manifest"), result=game.pop("result"),
                       trajectory=read_json(directory / "trajectory.json"), source_snapshot=freeze,
                       source_snapshot_reference="source-snapshot/ (reference only; source code not exported)",
                       artifact_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in directory.iterdir() if p.is_file() and p.name in {
                               "manifest.json", "result.json", "trajectory.json", "decisions.jsonl", "harness-trace.jsonl", "game.pgn"}}))
            manifest = read_json(directory / "manifest.json", {})
            result = read_json(directory / "result.json", {})
            game.update(id=game_id, run_id=run_id)
            write_json(game_dir / "index.json", game)
            outcome = result.get("replay_result") if result.get("completed") is True else None
            run["games"].append(dict(id=game_id, path=f"runs/{run_id}/{game_id}/index.json",
                event_count=len(game["events"]), completed=result.get("completed"),
                outcome=outcome, stop_reason=result.get("stop_reason"),
                policy_version=manifest.get("policy_version") or result.get("policy_version"),
                model=manifest.get("llm_model") or manifest.get("requested_model") or config.get("model"),
                opponent=manifest.get("opponent") or result.get("opponent") or config.get("opponent"),
                max_nodes=manifest.get("max_nodes") or config.get("max_nodes"),
                llm_token_usage=manifest.get("llm_token_usage"),
                llm_seconds=manifest.get("llm_seconds"), memory_used=manifest.get("memory_used"),
                search_version=manifest.get("search_version"), warnings=game["warnings"],
                highlights=[dict(id=e["id"], type=e["type"], turn=e["turn"]) for e in game["events"]
                            if e["type"] in {"verification", "replan", "error", "commit"}][:12]))
        write_json(target / "manifest.json", run)
        catalog["runs"].append(dict(id=run_id, title=run_id, path=f"runs/{run_id}/manifest.json"))
        if make_default:
            catalog["default_run"] = run_id
        write_json(catalog_path, catalog)
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        raise
    return run


def import_benchmark(source, output, make_default=False):
    """Import the four immediate gameplay groups, excluding archived source copies."""
    source, output = Path(source).resolve(), Path(output).resolve()
    groups = [(elo, policy, source / f"stockfish-{elo}" / policy)
              for elo in (1500, 2000) for policy in ("harness", "search")]
    for elo, policy, directory in groups:
        config = read_json(directory / "experiment-config.json", {})
        if config.get("policy") != policy or config.get("opponent", {}).get("elo") != elo:
            raise ValueError(f"Missing or mismatched benchmark group: stockfish-{elo}/{policy}")
    catalog = read_json(output / "manifest.json", dict(schema_version=SCHEMA_VERSION, runs=[]))
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported catalog schema version")
    ids = [f"benchmark-report-{elo}-{policy}" for elo, policy, _ in groups]
    if any((output / "runs" / run_id).exists() or any(r["id"] == run_id for r in catalog["runs"]) for run_id in ids):
        raise ValueError("Benchmark already imported; existing links are preserved")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Source and export directories must be separate")
    output.parent.mkdir(parents=True, exist_ok=True)
    moved = []
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        runs, entries = [], []
        for (elo, policy, directory), run_id in zip(groups, ids):
            run = import_run(directory, staging, run_id)
            run.update(title=f"{policy.title()} · {elo} Elo", collection="benchmark-report",
                       policy=policy, elo=elo, source_directory=f"{source.name}/stockfish-{elo}/{policy}",
                       selection=read_json(directory / "selection.json"))
            write_json(Path(staging) / "runs" / run_id / "manifest.json", run)
            entries.append({k: run[k] for k in ("id", "title", "collection", "policy", "elo")})
            entries[-1].update(path=f"runs/{run_id}/manifest.json")
            runs.append(run)
        try:
            (output / "runs").mkdir(parents=True, exist_ok=True)
            for run_id in ids:
                target = output / "runs" / run_id
                shutil.move(str(Path(staging) / "runs" / run_id), target)
                moved.append(target)
            catalog["runs"].extend(entries)
            if make_default:
                catalog["default_run"] = ids[0]
            write_json(output / "manifest.json", catalog)
        except Exception:
            for target in moved:
                shutil.rmtree(target)
            raise
    return runs


def export_reviewed(site, destination):
    site, destination = Path(site).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Export destination must not exist")
    if destination.is_relative_to(site) or site.is_relative_to(destination):
        raise ValueError("Export must be outside the source site")
    if not (site / "data/manifest.json").is_file():
        raise ValueError("Import a dataset before preparing an export")
    # Only known browser assets and the sanitized dataset enter the publication package.
    destination.mkdir(parents=True)
    for name in ("index.html", "viewer.html", "styles.css", "app.js", "model.js", "data"):
        path = site / name
        if path.is_symlink() or any(p.is_symlink() for p in path.rglob("*") if path.is_dir()):
            raise ValueError("Symlinks are not allowed in public exports")
        shutil.copytree(path, destination / name) if path.is_dir() else shutil.copy2(path, destination / name)
    return {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.rglob("*")) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("import")
    ingest.add_argument("source", type=Path)
    ingest.add_argument("--output", type=Path, default=Path("research_site/data"))
    ingest.add_argument("--run-id")
    ingest.add_argument("--games", nargs="+")
    ingest.add_argument("--default", action="store_true", help="Select this run when opening the viewer without a run link")
    benchmark = commands.add_parser("benchmark", help="Import benchmark-report's Search/Harness × 1500/2000 groups")
    benchmark.add_argument("source", type=Path)
    benchmark.add_argument("--output", type=Path, default=Path("research_site/data"))
    benchmark.add_argument("--default", action="store_true")
    publish = commands.add_parser("export")
    publish.add_argument("--site", type=Path, default=Path("research_site"))
    publish.add_argument("--output", type=Path, required=True)
    publish.add_argument("--reviewed", action="store_true", help="Confirm you reviewed every browser-accessible artifact")
    args = parser.parse_args()
    try:
        if args.command == "import":
            run = import_run(args.source, args.output, args.run_id, args.games, args.default)
            print(f"Imported {len(run['games'])} game(s) as {run['id']}. Public export still requires manual review.")
        elif args.command == "benchmark":
            runs = import_benchmark(args.source, args.output, args.default)
            print(f"Imported {sum(len(run['games']) for run in runs)} games across {len(runs)} benchmark groups. Nothing uploaded.")
        else:
            if not args.reviewed:
                parser.error("Review the sanitized data, raw views and metadata, then pass --reviewed. Nothing is uploaded.")
            hashes = export_reviewed(args.site, args.output)
            print(f"Prepared {len(hashes)} reviewed files locally. Nothing uploaded or deployed.")
            (args.output.parent / f"{args.output.name}-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Import/export failed: {exc}\n")


if __name__ == "__main__":
    main()
