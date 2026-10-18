"""Adapters for the recorded chess JSON/JSONL artifacts. No source is executed."""

from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import re

import chess
import chess.pgn

SCHEMA_VERSION = 1


def read_json(path, default=None):
    value = json.loads(path.read_text()) if path.is_file() else default
    if isinstance(default, dict) and not isinstance(value, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return value


def read_rows(path):
    rows, warnings = [], []
    if path.is_file():
        for line, text in enumerate(path.read_text().splitlines(), 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
                if not isinstance(row, dict):
                    raise ValueError("expected an object")
                rows.append((line, row))
            except ValueError:
                warnings.append(f"{path.name}:{line}: malformed record skipped")
    return rows, warnings


def request_context(record):
    for message in reversed(record.get("request") or []):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            try:
                data = json.loads(message.get("content", ""))
                if isinstance(data, dict):
                    return data.get("context") or {}, data.get("feedback") or {}
            except (ValueError, TypeError):
                pass
    return {}, {}


def valid_fen(fen):
    try:
        return fen if isinstance(fen, str) and chess.Board(fen).is_valid() else None
    except ValueError:
        return None


def advance(fen, uci):
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if not board.is_valid() or move not in board.legal_moves:
            return None
        board.push(move)
        return board.fen()
    except (ValueError, TypeError):
        return None


def load_game(directory: Path):
    """Return a versioned game plus lazy payloads, preserving source line order."""
    manifest = read_json(directory / "manifest.json", {})
    result = read_json(directory / "result.json", {})
    decisions, warnings = read_rows(directory / "decisions.jsonl")
    records, trace_warnings = read_rows(directory / "harness-trace.jsonl")
    warnings += trace_warnings
    adapter = "harness-jsonl-v1" if records else "decisions-jsonl-v1"
    groups = OrderedDict()
    for line, record in records:
        context, _ = request_context(record)
        key = record.get("position_id") or context.get("fen") or f"record-{line}"
        groups.setdefault(key, {"records": [], "decision": None})["records"].append((line, record))
    for line, decision in decisions:
        key = decision.get("position_id") or decision.get("root_fen") or f"decision-{line}"
        group = groups.setdefault(key, {"records": [], "decision": None})
        group["decision"] = (line, decision)
        if not records and decision.get("harness_records"):
            group["records"] = [(f"{line}.harness_records.{i}", r)
                                for i, r in enumerate(decision["harness_records"])]

    # FEN's move number provides a cross-file anchor; trace order breaks ties.
    def group_turn(group):
        pair = group["decision"]
        fen = (pair[1].get("root_fen") if pair else None)
        if not fen and group["records"]:
            fen = request_context(group["records"][0][1])[0].get("fen")
        return chess.Board(fen).fullmove_number if valid_fen(fen) else float("inf")

    events, payloads = [], {}

    def add(kind, title, source, location, evidence, fen=None, phase="before White move",
            board_origin="recorded", candidates=None, budget=None, move=None, turn=None, analysis=None):
        safe_kind = kind if re.fullmatch(r"[A-Za-z0-9_-]+", kind) else hashlib.sha256(kind.encode()).hexdigest()[:16]
        event_id = f"{source.replace('.', '-')}-{location}-{safe_kind}"
        fen = valid_fen(fen)
        event = dict(id=event_id, type=kind, title=title, turn=turn, fen=fen,
                     board_phase=phase if fen else "unavailable",
                     board_origin=board_origin if fen else "unavailable", move=move,
                     source=dict(file=source, location=str(location)),
                     timestamp=None, candidates=candidates or [], budget=budget or {},
                     analysis=analysis or {},
                     detail=f"events/{event_id}.json")
        event["search"] = json.dumps(evidence, ensure_ascii=False).lower()
        events.append(event)
        payloads[event_id] = dict(evidence=evidence)
        return event

    for turn_index, group in enumerate(sorted(groups.values(), key=group_turn), 1):
        decision_pair = group["decision"]
        decision = decision_pair[1] if decision_pair else {}
        root = valid_fen(decision.get("root_fen"))
        for line, record in group["records"]:
            source = "harness-trace.jsonl" if records else "decisions.jsonl"
            context, feedback = request_context(record)
            root = valid_fen(context.get("fen")) or root
            turn = chess.Board(root).fullmove_number if root else turn_index
            action = record.get("action") or {}
            output = record.get("execution_feedback")
            analyses = (output or {}).get("analyses") or []
            latest = analyses[-1] if analyses else {}
            base_budget = context.get("budget") or {}
            budget = dict(max_nodes=base_budget.get("max_nodes"),
                          remaining_nodes=feedback.get("remaining_nodes"),
                          remaining_llm_calls=feedback.get("remaining_llm_calls"))
            common = dict(fen=root, budget=budget, turn=turn)
            raw = dict(record=record)
            if record.get("request"):
                add("context", "Context preparation", source, line,
                    dict(messages=record["request"], **raw), **common)
            if record.get("response") is not None:
                response_event = add("llm", "Assistant response", source, line,
                    dict(response=record["response"], duration_seconds=record.get("seconds"), **raw), **common)
                created = record["response"].get("created")
                if type(created) in (int, float) and 0 <= created <= 253402300799:
                    response_event.update(timestamp=created, timestamp_origin="provider response.created; Unix seconds")
            if action:
                label = "Commit request" if action.get("action") == "commit" else "Search request"
                add("tool_call", label, source, line, dict(action=action, **raw), **common)
            if output is not None:
                kind = "verification" if latest.get("verification") else "tool_result"
                after_budget = dict(budget, remaining_nodes=output.get("remaining_nodes"),
                                    remaining_llm_calls=output.get("remaining_llm_calls"))
                add(kind, {"verification": "Verification feedback", "replan": "Replanning required"}.get(kind, "Search feedback"),
                    source, line, dict(feedback=output, **raw), fen=root, turn=turn,
                    budget=after_budget, candidates=latest.get("top_candidates"),
                    analysis={k: latest.get(k) for k in ("completed_depth", "candidate_scope", "score_scope")})
                if output.get("replan_required"):
                    add("replan", "Replanning required", source, f"{line}.replan",
                        dict(replan_events=output.get("replan_events"), **raw),
                        fen=root, turn=turn, budget=after_budget)
            if record.get("error") or record.get("status") == "rejected":
                add("error", "Action rejected / retry evidence", source, line,
                    dict(error=record.get("error"), status=record.get("status"), **raw), **common)
            if not decision_pair and record.get("status") == "accepted" and record.get("committed_move"):
                move = record["committed_move"]
                add("commit", f"White commits {move}", source, f"{line}.committed_move",
                    dict(committed_move=move, **raw), fen=advance(root, move) if root else None,
                    phase="after White move", board_origin="reconstructed from recorded root FEN + legal UCI",
                    move=move, turn=turn)
            if not any(record.get(k) for k in ("request", "response", "action", "execution_feedback", "error")):
                add(str(record.get("type") or "other"), "Additional recorded event", source, line, raw, **common)

        if decision_pair:
            line, decision = decision_pair
            turn = chess.Board(root).fullmove_number if root else turn_index
            if not group["records"]:
                if not decision.get("analysis_trace") and (decision.get("top_candidates") or decision.get("search_performed")):
                    add("tool_result", "Search result", "decisions.jsonl", f"{line}.search",
                        dict(decision=decision), fen=root, turn=turn,
                        candidates=decision.get("top_candidates"),
                        analysis={k: decision.get(k) for k in ("completed_depth", "score_scope")},
                        budget=dict(max_nodes=decision.get("max_nodes") or manifest.get("max_nodes"),
                                    nodes_used=decision.get("nodes_used")))
                for i, analysis in enumerate(decision.get("analysis_trace") or []):
                    evidence = analysis.get("evidence") or {}
                    add("verification" if analysis.get("verification") else "tool_result",
                        "Recorded search evidence", "decisions.jsonl", f"{line}.analysis_trace.{i}",
                        dict(analysis=analysis), fen=root, turn=turn,
                        candidates=evidence.get("top_candidates"),
                        analysis={k: evidence.get(k) for k in ("completed_depth", "score_scope")},
                        budget=dict(max_nodes=manifest.get("max_nodes"), remaining_nodes=analysis.get("remaining_nodes")))
            move = decision.get("committed_move") or decision.get("chosen_move")
            after_white = advance(root, move) if root and move else None
            add("commit", f"White commits {move or 'unavailable'}", "decisions.jsonl", line,
                dict(decision=decision), fen=after_white, turn=turn,
                phase="after White move", board_origin="reconstructed from recorded root FEN + legal UCI",
                move=move, budget=dict(max_nodes=manifest.get("max_nodes"), nodes_used=decision.get("nodes_used")))
            reply = decision.get("engine_move")
            if reply:
                after_black = advance(after_white, reply) if after_white else None
                recorded_post = valid_fen(decision.get("post_fen"))
                if recorded_post and after_black and recorded_post != after_black:
                    warnings.append(f"decisions.jsonl:{line}: post FEN disagrees with legal replay")
                add("opponent", f"Opponent replies {reply}", "decisions.jsonl", f"{line}.engine_move",
                    dict(engine_move=reply, post_fen=decision.get("post_fen")),
                    fen=recorded_post or after_black, turn=turn, phase="after opponent move",
                    board_origin="recorded" if recorded_post else "reconstructed from recorded moves", move=reply)

    if not events:
        history = result.get("history") or []
        initial = valid_fen(manifest.get("initial_fen"))
        if not history and (directory / "game.pgn").is_file():
            pgn = chess.pgn.read_game(io.StringIO((directory / "game.pgn").read_text()))
            if pgn and not pgn.errors:
                initial = pgn.board().fen()
                history = [dict(uci=m.uci()) for m in pgn.mainline_moves()]
        if history:
            adapter = "history-pgn-v1"
            fen = initial
            for i, row in enumerate(history, 1):
                fen = advance(fen, row.get("uci")) if fen else None
                add("move", f"Recorded move {row.get('uci', '?')}",
                    "result.json" if result.get("history") else "game.pgn", i,
                    dict(history=row), fen=fen, phase="after recorded move", turn=(i + 1) // 2,
                    board_origin="reconstructed from recorded initial FEN + legal moves", move=row.get("uci"))
            warnings.append("Move history only; individual harness events unavailable")
        elif valid_fen(result.get("fen")):
            add("state", "Recorded final position", "result.json", "fen", dict(result=result),
                fen=result["fen"], phase="final recorded position")
        else:
            warnings.append("No usable board or event evidence")

    # Keep search compact; unknown fields remain in lazy raw payloads.
    for event in events:
        evidence = payloads[event["id"]]["evidence"]
        searchable = {k: v for k, v in evidence.items() if k != "record"}
        event["search"] = json.dumps(searchable, ensure_ascii=False).lower()
    return dict(schema_version=SCHEMA_VERSION, adapter=adapter, events=events,
                manifest=manifest, result=result, warnings=warnings), payloads
