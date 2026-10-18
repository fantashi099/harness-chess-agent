"""Clearly synthetic fixtures exercising the recorded schemas, never experimental data."""

import hashlib
import json
from pathlib import Path

import chess
import pytest

from assignment.viewer.__main__ import export_reviewed, import_run, import_benchmark
from assignment.viewer.adapters import advance, load_game
from assignment.viewer.sanitize import sanitize


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def fixture_run(tmp_path, name="synthetic", older=False):
    source = tmp_path / name
    directory = source / "game_1"
    fen = chess.STARTING_FEN
    feedback = dict(remaining_nodes=70, analyses=[dict(verification=not older,
                    top_candidates=[dict(move="e2e4", primary_score=10, bound="exact", pv=["e2e4", "e7e5"])])])
    rows = [dict(position_id="p1", request=[dict(role="system", content="SYNTHETIC TEST PROMPT"),
                dict(role="user", content=json.dumps(dict(context=dict(fen=fen, budget=dict(max_nodes=100)),
                                                        feedback=dict(remaining_nodes=100))))],
                response=dict(choices=[dict(message=dict(content='{"action":"analyze"}'))]),
                action=dict(action="analyze", candidate_moves=[]), status="accepted", execution_feedback=feedback),
            dict(position_id="p1", action=dict(action="commit", move="e2e4"), status="accepted")]
    if older:
        rows[0].pop("response")
    directory.mkdir(parents=True)
    (directory / "harness-trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    after_white = advance(fen, "e2e4")
    after_black = advance(after_white, "e7e5")
    decision = dict(position_id="p1", root_fen=fen, committed_move="e2e4", chosen_move="e2e4",
                    engine_move="e7e5", post_fen=after_black, nodes_used=30)
    (directory / "decisions.jsonl").write_text(json.dumps(decision) + "\n")
    dump(directory / "manifest.json", dict(initial_fen=fen, policy_version="synthetic-old" if older else "synthetic-new", max_nodes=100))
    dump(directory / "result.json", dict(completed=False, stop_reason="test_only", history=[]))
    dump(source / "experiment-config.json", dict(model="SYNTHETIC TEST MODEL", api_key="secret-fixture-value"))
    return source, directory


def test_order_board_and_old_schema(tmp_path):
    for older in (False, True):
        _, directory = fixture_run(tmp_path, f"synthetic-{older}", older)
        game, _ = load_game(directory)
        assert [e["type"] for e in game["events"]] == (["context"] + ([] if older else ["llm"]) +
                ["tool_call", "tool_result" if older else "verification", "tool_call", "commit", "opponent"])
        context, commit, reply = game["events"][0], game["events"][-2], game["events"][-1]
        assert context["fen"] == chess.STARTING_FEN
        assert context["move"] is None
        assert chess.Board(commit["fen"]).piece_at(chess.E4).symbol() == "P"
        assert chess.Board(commit["fen"]).piece_at(chess.E7).symbol() == "p"
        assert chess.Board(reply["fen"]).piece_at(chess.E5).symbol() == "p"
        assert commit["board_phase"] == "after White move"
        assert context["board_origin"] == "recorded"
        assert commit["board_origin"].startswith("reconstructed")
        assert all(e["timestamp"] is None for e in game["events"])


def test_add_run_sanitizes_every_file_and_preserves_input(tmp_path):
    source, directory = fixture_run(tmp_path)
    input_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "sanitized"
    first = import_run(source, output)
    game_path = output / first["games"][0]["path"]
    before = game_path.read_bytes()
    second, _ = fixture_run(tmp_path, "synthetic-older", older=True)
    import_run(second, output, make_default=True)
    catalog = json.loads((output / "manifest.json").read_text())
    assert len(catalog["runs"]) == 2
    assert catalog["default_run"] == "synthetic-older"
    assert game_path.read_bytes() == before
    assert first["games"][0]["outcome"] is None
    assert first["games"][0]["completed"] is False
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in input_hashes.items())
    assert not any("secret-fixture-value" in p.read_text() for p in output.rglob("*.json"))
    with pytest.raises(ValueError, match="already imported"):
        import_run(source, output)


def test_missing_fields_partial_records_unknown_types_and_unsafe_type(tmp_path):
    directory = tmp_path / "partial"
    directory.mkdir()
    unfamiliar = {"type":"../../outside", "new_field":{"anything":True}}
    (directory / "harness-trace.jsonl").write_text(json.dumps(unfamiliar) + '\n{"partial":\n')
    game, payloads = load_game(directory)
    assert game["warnings"]
    assert game["events"][0]["fen"] is None
    assert game["events"][0]["type"] == "../../outside"
    assert "/" not in game["events"][0]["id"]
    assert next(iter(payloads.values()))["evidence"]["record"] == unfamiliar


def test_illegal_moves_and_missing_roots_do_not_invent_positions(tmp_path):
    assert advance(chess.STARTING_FEN, "e2e5") is None
    assert advance("garbage", "e2e4") is None
    _, directory = fixture_run(tmp_path)
    (directory / "harness-trace.jsonl").unlink()
    decision = dict(chosen_move="e2e4")
    (directory / "decisions.jsonl").write_text(json.dumps(decision))
    game, _ = load_game(directory)
    assert game["events"][0]["fen"] is None


def test_embedded_records_and_trace_only_commit(tmp_path):
    _, directory = fixture_run(tmp_path)
    records = [json.loads(line) for line in (directory / "harness-trace.jsonl").read_text().splitlines()]
    records[-1]["committed_move"] = "e2e4"
    (directory / "harness-trace.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    decision = json.loads((directory / "decisions.jsonl").read_text())
    (directory / "decisions.jsonl").unlink()
    game, _ = load_game(directory)
    assert game["events"][-1]["type"] == "commit"
    assert chess.Board(game["events"][-1]["fen"]).turn == chess.BLACK
    (directory / "harness-trace.jsonl").unlink()
    decision["harness_records"] = records
    (directory / "decisions.jsonl").write_text(json.dumps(decision))
    game, _ = load_game(directory)
    assert game["events"][0]["type"] == "context"
    assert game["events"][-1]["type"] == "opponent"


def test_replan_does_not_hide_verification(tmp_path):
    _, directory = fixture_run(tmp_path)
    records = [json.loads(line) for line in (directory / "harness-trace.jsonl").read_text().splitlines()]
    records[0]["execution_feedback"].update(replan_required=True, replan_events=[dict(move="e2e4")])
    (directory / "harness-trace.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    game, _ = load_game(directory)
    assert [e["type"] for e in game["events"]][3:5] == ["verification", "replan"]


def test_provider_time_is_only_attached_to_assistant_event(tmp_path):
    _, directory = fixture_run(tmp_path)
    records = [json.loads(line) for line in (directory / "harness-trace.jsonl").read_text().splitlines()]
    records[0]["response"]["created"] = 1700000000
    (directory / "harness-trace.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    game, _ = load_game(directory)
    assert game["events"][0]["timestamp"] is None
    assert game["events"][1]["timestamp"] == 1700000000
    assert game["events"][2]["timestamp"] is None


def test_history_and_pgn_fallback(tmp_path):
    directory = tmp_path / "history"
    directory.mkdir()
    dump(directory / "manifest.json", dict(initial_fen=chess.STARTING_FEN))
    dump(directory / "result.json", dict(history=[dict(uci="e2e4"), dict(uci="e7e5")]))
    game, _ = load_game(directory)
    assert game["adapter"] == "history-pgn-v1"
    assert len(game["events"]) == 2
    assert game["events"][-1]["fen"] == advance(advance(chess.STARTING_FEN, "e2e4"), "e7e5")
    (directory / "result.json").unlink()
    (directory / "game.pgn").write_text('[Result "*"]\n\n1. e4 e5 *\n')
    game, _ = load_game(directory)
    assert len(game["events"]) == 2


@pytest.mark.parametrize("value", [
    {"api_key":"NEVER_PUBLIC", "unknown":{"password":"NEVER_PUBLIC"}},
    json.dumps({"api_key":"NEVER_PUBLIC"}),
    "Authorization: Bearer NEVER_PUBLIC",
    "api_key=NEVER_PUBLIC",
    'set api_key = "NEVER_PUBLIC" in a config',
    'debug output {"password": "NEVER_PUBLIC"} appended',
    "sk-NEVER_PUBLIC123456789",
    "https://user:NEVER_PUBLIC@example.com/api?token=NEVER_PUBLIC",
    "/home/NEVER_PUBLIC/project/secrets.txt",
    "C:\\Users\\NEVER_PUBLIC\\secrets.txt",
    "NEVER_PUBLIC@example.com",
    "-----BEGIN PRIVATE KEY-----\nNEVER_PUBLIC\n-----END PRIVATE KEY-----",
    {"/home/NEVER_PUBLIC/path":"private path used as a key"},
])
def test_sanitization(value):
    assert "NEVER_PUBLIC" not in json.dumps(sanitize(value))


def test_recorded_fields_survive_sanitizing():
    evidence = dict(total_tokens=100, completion_tokens=20, position_id="abc", nodes_used=30,
                    tool_arguments={"candidate_moves":["e2e4"]}, future_field="<script>alert(1)</script>")
    assert sanitize(evidence) == evidence


def test_export_only_browser_files(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    for name in ["index.html", "viewer.html", "styles.css", "app.js", "model.js"]:
        (site / name).write_text("synthetic asset")
    dump(site / "data/manifest.json", dict(schema_version=1, runs=[]))
    (site / "blog.js").write_text("ARCHIVED_BLOG_ONLY")
    (site / "blog.css").write_text("ARCHIVED_BLOG_ONLY")
    (site / ".env").write_text("NEVER_PUBLIC")
    (site / "README.md").write_text("NEVER_PUBLIC")
    dest = tmp_path / "reviewed"
    hashes = export_reviewed(site, dest)
    assert ".env" not in hashes and "README.md" not in hashes
    assert set(hashes) == {"index.html", "viewer.html", "styles.css", "app.js", "model.js", "data/manifest.json"}


def test_no_supported_files_or_bad_id(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No supported"):
        import_run(empty, tmp_path / "output")
    with pytest.raises(ValueError, match="Run ID"):
        import_run(empty, tmp_path / "output", run_id="../escape")


def test_pure_search_keeps_root_candidates_budget_and_raw_decision(tmp_path):
    _, directory = fixture_run(tmp_path)
    (directory / "harness-trace.jsonl").unlink()
    decision = json.loads((directory / "decisions.jsonl").read_text())
    decision.update(search_performed=True, max_nodes=100, completed_depth=4,
                    top_candidates=[dict(move="e2e4", primary_score=52, pv=["e2e4"])],
                    reason="SYNTHETIC SEARCH REASON", unknown_field=True)
    (directory / "decisions.jsonl").write_text(json.dumps(decision))
    game, payloads = load_game(directory)
    assert [event["type"] for event in game["events"]] == ["tool_result", "commit", "opponent"]
    search = game["events"][0]
    assert search["fen"] == chess.STARTING_FEN
    assert search["budget"] == dict(max_nodes=100, nodes_used=30)
    assert search["analysis"]["completed_depth"] == 4
    assert search["candidates"] == decision["top_candidates"]
    assert payloads[search["id"]]["evidence"]["decision"] == decision


def test_benchmark_groups_order_selection_and_atomic_failure(tmp_path):
    root = tmp_path / "benchmark-report"
    output = tmp_path / "viewer-data"
    old, _ = fixture_run(tmp_path, "existing")
    import_run(old, output, make_default=True)
    for elo in (1500, 2000):
        for policy in ("search", "harness"):
            source, directory = fixture_run(root / f"stockfish-{elo}", policy)
            dump(source / "experiment-config.json", dict(policy=policy, opponent=dict(elo=elo)))
            for number in (2, 10):
                game_dir = source / f"game_{number}"
                dump(game_dir / "result.json", dict(completed=True, replay_result="1-0"))
            # Nested archives are not additional games.
            dump(source / "sources/old/game_99/result.json", dict(completed=True))
    curated = root / "stockfish-2000/harness"
    dump(curated / "selection.json", dict(random_sample=False, note="SYNTHETIC SELECTION"))
    bad = curated / "game_2/result.json"
    original = bad.read_text()
    bad.write_text("invalid json")
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(ValueError):
        import_benchmark(root, output, True)
    assert (output / "manifest.json").read_bytes() == before
    assert sorted(p.name for p in (output / "runs").iterdir()) == ["existing"]
    bad.write_text(original)
    runs = import_benchmark(root, output, True)
    assert len(runs) == 4
    assert all([game["id"] for game in run["games"]] == ["game_1", "game_2", "game_10"] for run in runs)
    curated_run = next(run for run in runs if run["id"] == "benchmark-report-2000-harness")
    assert curated_run["selection"]["random_sample"] is False
    assert curated_run["source_directory"] == "benchmark-report/stockfish-2000/harness"
    catalog = json.loads((output / "manifest.json").read_text())
    assert len(catalog["runs"]) == 5
    assert catalog["default_run"] == "benchmark-report-1500-harness"
    assert [(entry["policy"], entry["elo"]) for entry in catalog["runs"][1:]] == [("harness",1500),("search",1500),("harness",2000),("search",2000)]
    with pytest.raises(ValueError, match="already imported"):
        import_benchmark(root, output)
