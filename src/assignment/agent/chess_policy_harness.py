"""Read-only v5 analysis tools; an explicit LLM action owns the final move."""
from copy import deepcopy
import hashlib
import json
from time import monotonic

import chess

if __package__:
    from assignment.agent import chess_policy_search as search
else:
    import chess_policy_search as search

POLICY_VERSION = "select-move-v6-llm-harness-13"
REVISION = 13
EVALUATION = "development-king-safety-v2"
MAX_ANALYSES = 4
MAX_LLM_CALLS = 8
SCORE_DROP_THRESHOLD = 150
TACTICAL_PROBE_NODES = 500


def board_facts(board):
    return {"pieces": {chess.square_name(s): p.symbol() for s, p in sorted(board.piece_map().items())},
            "bishop_square_colors": {chess.square_name(s): "light" if
                (chess.square_file(s)+chess.square_rank(s)) % 2 else "dark"
                for color in (chess.WHITE, chess.BLACK) for s in board.pieces(chess.BISHOP, color)},
            "in_check": board.is_check(),
            "undeveloped_white_minors": [chess.square_name(s) for s in search.undeveloped_minors(board, chess.WHITE)],
            "legal_white_castling": [m.uci() for m in board.legal_moves if board.is_castling(m)],
            "white_castling_rights": {"kingside": board.has_kingside_castling_rights(chess.WHITE),
                                      "queenside": board.has_queenside_castling_rights(chess.WHITE)},
            "legal_development_moves": [m.uci() for m in board.legal_moves
                if m.from_square in search.undeveloped_minors(board, chess.WHITE)],
            "development_scope": "Home-square heuristic; a returned minor can appear undeveloped. Castling list is legal now, rights alone do not imply a clear path."}


def move_facts(board, text):
    move = chess.Move.from_uci(text)
    captured_square = move.to_square
    if board.is_en_passant(move):
        captured_square += -8 if board.turn else 8
    captured = board.piece_at(captured_square) if board.is_capture(move) else None
    facts = {"move": text, "piece": board.piece_at(move.from_square).symbol(), "san": board.san(move),
             "captured_piece": captured.symbol() if captured else None,
             "captured_square": chess.square_name(captured_square) if captured else None}
    snapshot = board.copy(stack=True)
    snapshot.push(move)
    facts.update(gives_check=snapshot.is_check(),
                 geometric_defenders_after=sorted(chess.square_name(s) for s in snapshot.attackers(board.turn, move.to_square)),
                 opponent_captures_after=[{"move": m.uci(), "san": snapshot.san(m),
                     "captured_piece": (snapshot.piece_at(m.to_square).symbol()
                         if not snapshot.is_en_passant(m) else ("P" if board.turn else "p"))}
                     for m in snapshot.legal_moves if snapshot.is_capture(m)],
                 attacks_after=sorted(chess.square_name(s) for s in snapshot.attacks(move.to_square)),
                 attack_scope="Geometric attacks by the moved piece after this move, including defended own pieces; not a proof of legal continuations")
    return facts


def prepare(board, position_id, max_nodes, max_depth, previous=None, opponent=None):
    return {"context": {"position_id": position_id, "fen": board.fen(), "board": str(board),
        "board_facts": board_facts(board),
        "opponent": opponent,
        "legal_moves": sorted(m.uci() for m in board.legal_moves),
        "recent_moves": [m.uci() for m in board.move_stack[-12:]],
        "actual_black_reply": board.peek().uci() if board.move_stack else None,
        "previous_decision": previous, "budget": {"max_nodes": max_nodes, "max_depth": max_depth,
            "max_analyses": MAX_ANALYSES, "max_llm_calls": MAX_LLM_CALLS,
            "score_drop_threshold": SCORE_DROP_THRESHOLD},
        "evaluator": EVALUATION},
        "analyses": [], "nodes_used": 0, "max_nodes": max_nodes,
        "verification_reserve": max(1, max_nodes//3), "comparison_reserve": max(1, max_nodes//6)}


def mate_safety(board, move, max_nodes):
    """Check the selected move; allow unavoidable mate only after checking alternatives."""
    snapshot = board.copy(stack=True)
    result = {"move": move, "complete": False, "mating_reply": None,
              "safe_alternative": None, "forced_mate": False, "nodes": 0,
              "checking_replies": [], "scope": "All immediate Black replies checked for mate in one only"}

    def push(candidate):
        if result["nodes"] >= max_nodes:
            raise search.BudgetExhausted
        result["nodes"] += 1
        snapshot.push(candidate)

    def mating_reply(candidate):
        push(candidate)
        try:
            if snapshot.outcome(claim_draw=True) is not None:
                return None
            for reply in list(snapshot.legal_moves):
                push(reply)
                try:
                    if candidate.uci() == move and snapshot.is_check():
                        result["checking_replies"].append(reply.uci())
                    if snapshot.is_checkmate():
                        return reply.uci()
                finally:
                    snapshot.pop()
            return None
        finally:
            snapshot.pop()

    try:
        result["mating_reply"] = mating_reply(chess.Move.from_uci(move))
        if result["mating_reply"]:
            for candidate in sorted(board.legal_moves, key=lambda m: m.uci()):
                if candidate.uci() != move and mating_reply(candidate) is None:
                    result["safe_alternative"] = candidate.uci()
                    break
            result["forced_mate"] = result["safe_alternative"] is None
        result["complete"] = True
    except search.BudgetExhausted:
        pass
    return result


def checking_mate_refutation(board, move, checking_replies, max_nodes):
    """Prove a checking Black reply mates against every legal White defense."""
    snapshot = board.copy(stack=True)
    result = {"status": "unknown", "nodes": 0, "black_reply": None, "defenses": [],
              "scope": "Checking Black replies, every White defense, then Black mate in one; quiet replies and longer mates not covered"}

    def push(candidate):
        if result["nodes"] >= max_nodes:
            raise search.BudgetExhausted
        result["nodes"] += 1
        snapshot.push(candidate)

    if not checking_replies:
        result["status"] = "not_found"
        return result
    try:
        push(chess.Move.from_uci(move))
        for text in sorted(checking_replies):
            reply = chess.Move.from_uci(text)
            push(reply)
            try:
                if snapshot.outcome(claim_draw=True) is not None:
                    continue
                defenses = list(snapshot.legal_moves)
                certificate = []
                for defense in defenses:
                    push(defense)
                    mating_reply = None
                    try:
                        if snapshot.outcome(claim_draw=True) is None:
                            for last in list(snapshot.legal_moves):
                                push(last)
                                try:
                                    if snapshot.is_checkmate():
                                        mating_reply = last.uci()
                                        break
                                finally:
                                    snapshot.pop()
                    finally:
                        snapshot.pop()
                    if mating_reply is None:
                        break
                    certificate.append({"white_defense": defense.uci(), "black_mate": mating_reply})
                if defenses and len(certificate) == len(defenses):
                    result.update(status="refuted", black_reply=text, defenses=certificate)
                    return result
            finally:
                snapshot.pop()
        result["status"] = "not_found"
    except search.BudgetExhausted:
        pass
    return result


def checking_chain_refutation(board, move, checking_replies, max_nodes):
    """Bounded AND/OR proof: Black checks, White may use every legal evasion."""
    snapshot = board.copy(stack=True)
    result = {"status": "unknown", "nodes": 0, "black_reply": None, "defenses": [],
              "mate_in": None, "scope": "Continuous Black checks up to mate in three; every White evasion covered. Quiet threats and longer mates not covered"}

    def push(candidate):
        if result["nodes"] >= max_nodes:
            raise search.BudgetExhausted
        result["nodes"] += 1
        snapshot.push(candidate)

    def prove(left, replies=None):
        if snapshot.outcome(claim_draw=True) is not None:
            return None
        checks = (list(snapshot.generate_legal_moves()) if replies is None else
                  [chess.Move.from_uci(text) for text in replies])
        checks = sorted((m for m in checks if snapshot.gives_check(m)),
                        key=lambda m: (not snapshot.is_capture(m), m.uci()))
        for reply in checks:
            push(reply)
            try:
                if snapshot.is_checkmate():
                    return {"black_mate": reply.uci()}, 1
                if left == 1 or snapshot.outcome(claim_draw=True) is not None:
                    continue
                defenses = list(snapshot.legal_moves)
                certificate = []; distance = 1
                for defense in defenses:
                    push(defense)
                    try:
                        continuation = prove(left-1)
                    finally:
                        snapshot.pop()
                    if continuation is None:
                        break
                    tree, length = continuation
                    certificate.append(dict(white_defense=defense.uci(), **tree))
                    distance = max(distance, 1+length)
                if defenses and len(certificate) == len(defenses):
                    return {"black_reply": reply.uci(), "defenses": certificate}, distance
            finally:
                snapshot.pop()
        return None

    if not checking_replies:
        result["status"] = "not_found"
        return result
    try:
        push(chess.Move.from_uci(move))
        proof = prove(3, checking_replies)
        if proof:
            tree, distance = proof
            result.update(status="refuted", black_reply=tree.get("black_reply", tree.get("black_mate")),
                          defenses=tree.get("defenses", []), mate_in=distance)
        else:
            result["status"] = "not_found"
    except search.BudgetExhausted:
        pass
    return result


def validate_action(action, context):
    if (not isinstance(action, dict) or not isinstance(action.get("action"), str)
            or action["action"] not in {"analyze", "commit"}):
        raise ValueError("Action must be analyze or commit")
    fields = {"action", "objective", "rationale"}
    fields |= {"candidate_moves", "max_nodes", "max_depth"} if action["action"] == "analyze" else {"move"}
    if action["action"] == "commit" and "continuation" in action:
        fields.add("continuation")
    if action["action"] == "commit" and "accept_risk" in action:
        fields.add("accept_risk")
        if type(action["accept_risk"]) is not bool:
            raise ValueError("accept_risk must be boolean")
    if set(action) != fields:
        raise ValueError(f"Unexpected or missing action fields; required: {sorted(fields)}; "
                         f"missing: {sorted(fields-set(action))}; unexpected: {sorted(set(action)-fields)}")
    for key, limit in (("objective", 320), ("rationale", 2000)):
        if not isinstance(action[key], str) or not 1 <= len(action[key]) <= limit:
            raise ValueError(f"Invalid {key}")
    if action["action"] == "commit":
        if action["move"] not in context["legal_moves"]:
            raise ValueError("Commit requires a legal UCI move")
        line = action.get("continuation", [action["move"]])
        if (not isinstance(line, list) or not 1 <= len(line) <= 8
                or any(not isinstance(move, str) or len(move) not in (4, 5) for move in line)
                or line[0] != action["move"]):
            raise ValueError("continuation must contain 1..8 UCI moves starting with the committed move")
    else:
        candidates = action["candidate_moves"]
        if (not isinstance(candidates, list) or len(candidates) > 5
                or any(not isinstance(m, str) or m not in context["legal_moves"] for m in candidates)
                or len(set(candidates)) != len(candidates)):
            raise ValueError("Use [] for all moves, or 1..5 unique legal candidates")
        for key, maximum in (("max_nodes", context["budget"]["max_nodes"]), ("max_depth", context["budget"]["max_depth"])):
            if type(action[key]) is not int or not 1 <= action[key] <= maximum:
                raise ValueError(f"Invalid {key}")
    return deepcopy(action)


def verification_reservation(session):
    if not session["analyses"] or len(session["context"]["legal_moves"]) == 1:
        return None
    move = session["analyses"][0]["evidence"]["chosen_move"]
    latest = next((t for t in reversed(session["analyses"]) if t["safety"] and t["safety"]["move"] == move), None)
    return None if latest and verification_usable(latest, session["context"]["budget"]["max_depth"]) else move


def node_allowance(session, verification=False, move=None):
    remaining = session["max_nodes"] - session["nodes_used"]
    if not verification:
        return max(0, min(remaining-session["verification_reserve"], session["max_nodes"]//2))
    reserve = 0
    if len(session["context"]["legal_moves"]) > 1:
        protected = verification_reservation(session)
        if (not any(t["verification"] for t in session["analyses"])
                or (protected and move != protected)):
            reserve = session["comparison_reserve"]
    return max(0, remaining-reserve)


def analyze(board, game_id, session, action):
    action = validate_action(action, session["context"])
    if action["action"] != "analyze":
        raise ValueError("Analysis requires analyze action")
    action_id = hashlib.sha256(json.dumps(action, sort_keys=True).encode()).hexdigest()
    for old in session["analyses"]:
        if old["action_id"] == action_id:
            return deepcopy(old)
    if len(session["analyses"]) >= MAX_ANALYSES:
        raise ValueError("Analysis allowance exhausted; choose from existing evidence")
    legal = set(session["context"]["legal_moves"])
    global_completed = any((not t["action"]["candidate_moves"] or set(t["action"]["candidate_moves"]) == legal)
                           and t["evidence"]["completed_depth"] >= 1
                           for t in session["analyses"])
    sole_move = len(session["context"]["legal_moves"]) == 1
    covers_all = not action["candidate_moves"] or set(action["candidate_moves"]) == legal
    if not covers_all and not global_completed and not sole_move:
        raise ValueError("Start with candidate_moves=[] (or every legal root move) and complete at least depth 1 over all legal moves")
    remaining = session["max_nodes"] - session["nodes_used"]
    if action["max_nodes"] > remaining:
        raise ValueError(f"Only {remaining} nodes remain")
    verification = len(action["candidate_moves"]) == 1 or sole_move
    selected = (action["candidate_moves"] or session["context"]["legal_moves"])[0]
    available = node_allowance(session, verification, selected)
    if action["max_nodes"] > available:
        raise ValueError(f"Only {available} analysis nodes available; reserve {session['verification_reserve']} for verification "
                         f"including {session['comparison_reserve']} for a possible alternative comparison. "
                         f"Unverified search suggestion: {verification_reservation(session)}; "
                         "check reserved_verification and commit_ready_moves in feedback")
    started = monotonic()
    safety = mate_safety(board, selected, action["max_nodes"]-1) if verification else None
    if safety and safety["complete"] and not safety["mating_reply"] and selected not in proven_refutations(session):
        probe_cap = min(TACTICAL_PROBE_NODES, action["max_nodes"]-safety["nodes"]-1)
        safety["tactical_refutation"] = checking_mate_refutation(board, selected, safety["checking_replies"], probe_cap)
        safety["nodes"] += safety["tactical_refutation"]["nodes"]
        # shortcut: extend king moves/check evasions only; broaden if quiet-threat regressions justify it.
        if (safety["tactical_refutation"]["status"] == "not_found"
                and (board.is_check() or board.piece_at(chess.Move.from_uci(selected).from_square).piece_type == chess.KING)):
            cap = probe_cap-safety["tactical_refutation"]["nodes"]
            safety["checking_chain_refutation"] = checking_chain_refutation(board, selected, safety["checking_replies"], cap)
            safety["nodes"] += safety["checking_chain_refutation"]["nodes"]
    if safety and safety["complete"] and (safety["mating_reply"] or
            safety.get("tactical_refutation", {}).get("status") == "refuted" or
            safety.get("checking_chain_refutation", {}).get("status") == "refuted"):
        uncovered = legal - proven_refutations(session).keys() - {selected}
        # shortcut: close one missing immediate-mate proof; larger/longer closures need more analysis.
        if len(uncovered) == 1:
            cap = min(TACTICAL_PROBE_NODES, action["max_nodes"]-safety["nodes"]-1)
            safety["remaining_root_safety"] = mate_safety(board, next(iter(uncovered)), cap)
            safety["nodes"] += safety["remaining_root_safety"]["nodes"]
    guard_nodes = safety["nodes"] if safety else 0
    effective_depth = session["context"]["budget"]["max_depth"] if verification else action["max_depth"]
    evidence = search.select_move(board, game_id, action["max_nodes"]-guard_nodes, effective_depth,
                                  action["candidate_moves"] or None, evaluation=EVALUATION)
    session["nodes_used"] += evidence["nodes_used"] + guard_nodes
    trace = {"action_id": action_id, "action": action, "evidence": evidence,
             "move_facts": move_facts(board, selected) if verification else None,
             "safety": safety,
             "verification": verification, "effective_max_depth": effective_depth,
             "remaining_nodes": session["max_nodes"] - session["nodes_used"],
             "search_seconds": monotonic() - started}
    session["analyses"].append(trace)
    return deepcopy(trace)


def exact_evidence(session):
    """A later shallow verification must not erase deeper exact search evidence."""
    best = {}
    for trace in session["analyses"]:
        for row in trace["evidence"]["top_candidates"]:
            if row["bound"] != "exact" or trace["evidence"]["completed_depth"] < 1:
                continue
            old = best.get(row["move"])
            if old is None or trace["evidence"]["completed_depth"] >= old[0]["evidence"]["completed_depth"]:
                best[row["move"]] = (trace, row)
    return best


def verified_iterations(session):
    """Completed single-candidate iterations use a full window at every depth."""
    rows = {}
    minimum = min(2, session["context"]["budget"]["max_depth"])
    for trace in session["analyses"]:
        if not verification_usable(trace, session["context"]["budget"]["max_depth"]):
            continue
        move = trace["safety"]["move"]
        for iteration in trace["evidence"]["iterations"]:
            if iteration["completed"] and iteration["depth"] >= minimum and iteration["move"] == move:
                rows.setdefault(move, {})[iteration["depth"]] = {
                    "score": iteration["primary_score"], "pv": iteration["pv"], "action_id": trace["action_id"]}
    return rows


def candidate_comparisons(session):
    iterations = verified_iterations(session)
    strongest = exact_evidence(session)
    moves = sorted(iterations)
    comparisons = []
    for index, move in enumerate(moves):
        for alternative in moves[index+1:]:
            shared = iterations[move].keys() & iterations[alternative].keys()
            if shared:
                depth = max(shared)
                comparisons.append({"moves": [move, alternative], "depth": depth,
                    "evidence": {m: iterations[m][depth] for m in (move, alternative)},
                    "depth_limited": any(depth < strongest[m][0]["evidence"]["completed_depth"]
                                         for m in (move, alternative))})
    return comparisons


def proven_refutations(session):
    proofs = {}
    for trace in session["analyses"]:
        safety = trace["safety"]
        if not safety:
            continue
        remaining = safety.get("remaining_root_safety")
        guards = [safety] + ([remaining] if remaining and remaining["complete"] else [])
        for guard in guards:
            tactical = guard.get("tactical_refutation", {})
            chain = guard.get("checking_chain_refutation", {})
            if guard["mating_reply"]:
                proof = {"mate_in": 1, "black_reply": guard["mating_reply"], "defenses": []}
            elif tactical.get("status") == "refuted":
                proof = {"mate_in": 2, "black_reply": tactical["black_reply"], "defenses": tactical["defenses"]}
            elif chain.get("status") == "refuted":
                proof = {"mate_in": chain["mate_in"], "black_reply": chain["black_reply"], "defenses": chain["defenses"]}
            else:
                continue
            move = guard["move"]
            if move not in proofs or proof["mate_in"] < proofs[move]["mate_in"]:
                proofs[move] = dict(proof, action_id=trace["action_id"])
    return proofs


def unavoidable_mate(session):
    return (any(t["safety"] and t["safety"]["forced_mate"] for t in session["analyses"])
            or set(session["context"]["legal_moves"]) <= proven_refutations(session).keys())


def replan_events(session):
    if not session["analyses"] or len(session["context"]["legal_moves"]) == 1:
        return []
    baseline = {r["move"]: r for r in session["analyses"][0]["evidence"]["top_candidates"] if r["bound"] == "exact"}
    best = exact_evidence(session)
    verified = {t["safety"]["move"] for t in session["analyses"] if verification_usable(t, session["context"]["budget"]["max_depth"])}
    pairs = candidate_comparisons(session)
    events = []
    for move in sorted(verified & baseline.keys() & best.keys()):
        trace, row = best[move]
        drop = baseline[move]["primary_score"] - row["primary_score"]
        if drop >= SCORE_DROP_THRESHOLD:
            depth = trace["evidence"]["completed_depth"]
            comparisons = []
            for pair in pairs:
                if move in pair["moves"]:
                    alternative = next(m for m in pair["moves"] if m != move)
                    comparisons.append({"move": alternative, "depth": pair["depth"],
                                        "selected": pair["evidence"][move],
                                        "alternative": pair["evidence"][alternative],
                                        "depth_limited": pair["depth"] < depth})
            events.append({"move": move, "initial_score": baseline[move]["primary_score"],
                           "verified_score": row["primary_score"], "drop": drop,
                           "required_comparison_depth": min(2, session["context"]["budget"]["max_depth"]),
                           "preferred_comparison_depth": depth, "comparisons": comparisons,
                           "compared_moves": [row["move"] for row in comparisons]})
    return events


def feedback(session):
    events = replan_events(session)
    pending = any(not event["compared_moves"] for event in events) and not unavoidable_mate(session)
    rows = []
    for trace in session["analyses"]:
        evidence = trace["evidence"]
        rows.append({"action_id": trace["action_id"], "objective": trace["action"]["objective"],
            "candidate_scope": trace["action"]["candidate_moves"] or "all legal moves",
            "suggested_move": evidence["chosen_move"], "primary_score": evidence["primary_score"],
            "completed_depth": evidence["completed_depth"], "pv": evidence["pv"],
            "top_candidates": evidence["top_candidates"], "cutoff_reached": evidence["cutoff_reached"],
            "nodes_used": evidence["nodes_used"] + (trace["safety"] or {}).get("nodes", 0),
            "quiescence": evidence["quiescence"],
            "score_scope": evidence["score_scope"], "safety": trace["safety"],
            "verification": trace["verification"], "effective_max_depth": trace["effective_max_depth"],
            "move_facts": trace["move_facts"],
            "commit_ready": commit_ready(trace, session) and not pending})
    strongest = {move: {"score": row["primary_score"], "depth": trace["evidence"]["completed_depth"],
                        "pv": row["pv"], "source_action_id": trace["action_id"]}
                 for move, (trace, row) in exact_evidence(session).items()}
    latest = {t["safety"]["move"]: t for t in session["analyses"] if t["safety"]}
    ready = sorted(move for move, trace in latest.items()
                   if commit_ready(trace, session) and not pending)
    reserved = verification_reservation(session)
    return {"analyses": rows, "remaining_nodes": session["max_nodes"] - session["nodes_used"],
            "broad_search_required": not any(t["evidence"]["completed_depth"] >= 1 and
                (not t["action"]["candidate_moves"] or set(t["action"]["candidate_moves"]) == set(session["context"]["legal_moves"]))
                for t in session["analyses"]),
            "remaining_analyses": MAX_ANALYSES - len(rows),
            "verification_reserve": session["verification_reserve"],
            "comparison_reserve": session["comparison_reserve"], "replan_required": pending,
            "replan_events": events, "strongest_exact_evidence": strongest,
            "candidate_comparisons": candidate_comparisons(session),
            "proven_refutations": proven_refutations(session), "unavoidable_mate": unavoidable_mate(session),
            "commit_ready_moves": ready,
            "risk_commit_moves": risk_commit_moves(session, ready),
            "reserved_verification": ({"move": reserved, "max_nodes": node_allowance(session, True, reserved)}
                                      if reserved else None),
            "analysis_nodes_available": node_allowance(session),
            "verification_nodes_available": node_allowance(session, True)}


def risk_commit_moves(session, ready):
    """An explicit LLM risk acceptance can finish a budget-exhausted turn."""
    if ready or not session["analyses"] or (session["nodes_used"] < session["max_nodes"]
            and len(session["analyses"]) < MAX_ANALYSES):
        return []
    latest = {t["safety"]["move"]: t for t in session["analyses"] if t["safety"]}
    candidates = sorted(move for move, trace in latest.items()
                        if trace["evidence"]["completed_depth"] >= 1
                        and move in exact_evidence(session))
    unrefuted = [m for m in candidates if m not in proven_refutations(session)]
    return unrefuted or candidates


def verification_usable(trace, max_depth):
    safety = trace["safety"]
    evidence = trace["evidence"]
    sufficient_depth = (evidence["completed_depth"] >= min(2, max_depth)
                        or evidence["primary_score"] == search.MATE-1)
    return bool(safety and safety["complete"] and sufficient_depth
                and evidence["top_candidates"] and evidence["top_candidates"][0]["bound"] == "exact")


def commit_ready(trace, session):
    return (verification_usable(trace, session["context"]["budget"]["max_depth"])
            and (unavoidable_mate(session) or
                 (not trace["safety"]["safe_alternative"] and trace["safety"]["move"] not in proven_refutations(session))))


def check_continuation(board, action, strict=True):
    snapshot = board.copy(stack=True)
    def material():
        return sum(search.VALUES[p.symbol().upper()] * (1 if p.color else -1) for p in snapshot.piece_map().values())
    initial = material()
    ledger = []
    error = None
    for ply, text in enumerate(action.get("continuation", [action["move"]]), 1):
        try:
            if snapshot.outcome(claim_draw=True) is not None:
                raise ValueError(f"Continuation continues after terminal position at ply {ply}")
            try:
                move = chess.Move.from_uci(text)
            except ValueError as exc:
                raise ValueError(f"Invalid UCI continuation at ply {ply}: {text}") from exc
            if move not in snapshot.legal_moves:
                raise ValueError(f"Illegal continuation at ply {ply}: {text}; check the actual pieces and squares")
        except ValueError as exc:
            if strict or ply == 1:
                raise
            error = str(exc)
            break
        san = snapshot.san(move)
        snapshot.push(move)
        ledger.append({"uci": text, "san": san, "white_material_balance": material(),
                       "change_from_root": material()-initial})
    outcome = snapshot.outcome(claim_draw=True)
    if strict and len(ledger) < 3 and outcome is None and board.legal_moves.count() > 1:
        raise ValueError("Give at least White move, expected Black reply and White continuation (3 plies), unless terminal")
    return {"valid": error is None, "error": error, "plies": ledger, "final_fen": snapshot.fen(), "terminal_result": outcome.result() if outcome else None,
            "scope": "Legal illustrative continuation and material accounting; not a prediction or proof of best defense"}


def decision_from(session, action, board, game_id):
    action = validate_action(action, session["context"])
    if action["action"] != "commit" or not session["analyses"]:
        raise ValueError("An explicit commit after analysis is required; no automatic policy fallback")
    verifications = [t for t in session["analyses"] if t["safety"] and t["safety"]["move"] == action["move"]]
    if not verifications:
        raise ValueError("Before commit, analyze ONLY the selected move for full-window verification and mate safety")
    verification = verifications[-1]
    proofs = proven_refutations(session)
    forced = unavoidable_mate(session)
    proof = proofs.get(action["move"])
    risk = action.get("accept_risk", False)
    if risk and action["move"] not in feedback(session)["risk_commit_moves"]:
        raise ValueError("Risk acceptance requires exhausted analysis budget and an eligible analyzed move")
    if proof and not forced and not risk:
        raise ValueError(f"Unsafe {action['move']}: Black has {proof['black_reply']} mate in {proof['mate_in']}; "
                         f"all White defenses: {proof['defenses']}. Analyze a different move and choose again")
    if not risk and not commit_ready(verification, session):
        raise ValueError("Before commit, analyze ONLY the selected move with completed depth >=2 (or configured max depth 1, or proven mate in one); "
                         "a complete mate safety check and exact score are required")
    events = replan_events(session)
    if any(not event["compared_moves"] for event in events) and not forced and not risk:
        raise ValueError(f"Replan required after score drop: {events}. Verify a DIFFERENT candidate at comparable depth before choosing again")
    continuation = check_continuation(board, action, strict=False)
    trace, row = exact_evidence(session)[action["move"]]
    evidence = trace["evidence"]
    first = session["analyses"][0]["evidence"]["chosen_move"]
    return {"policy_version": POLICY_VERSION, "root_fen": board.fen(), "chosen_move": action["move"],
        "executor_position_id": search.identity(board, game_id), "primary_score": row["primary_score"],
        "score_bound": row.get("bound"), "pv": row["pv"], "secondary": row.get("secondary", {}),
        "completed_depth": evidence["completed_depth"], "nodes_used": session["nodes_used"],
        "simulator_calls": session["nodes_used"], "primary_ties": [],
        "cutoff_reached": any(t["evidence"]["cutoff_reached"] for t in session["analyses"]),
        "reason": "explicit_llm_commit", "llm_action": action, "analysis_trace": deepcopy(session["analyses"]),
        "safety": deepcopy(verification["safety"]),
        "verification_nodes": verification["evidence"]["nodes_used"] + verification["safety"]["nodes"],
        "score_source_action_id": trace["action_id"], "verification_action_id": verification["action_id"],
        "replan_events": events, "candidate_comparisons": candidate_comparisons(session),
        "proven_refutations": proofs, "unavoidable_mate": forced,
        "risk_acceptance": ({"reason": "analysis_budget_exhausted", "known_refutation": proof,
            "unresolved_replan": any(not e["compared_moves"] for e in events),
            "safety_complete": verification["safety"]["complete"]} if risk else None),
        "continuation_checks": continuation,
        "verification_count": sum(t["verification"] for t in session["analyses"]),
        "comparison_count": max(0, len({t["safety"]["move"] for t in session["analyses"] if t["verification"]})-1),
        "replanned": (sum(not t["verification"] for t in session["analyses"]) > 1 or
                      len({t["safety"]["move"] for t in session["analyses"] if t["verification"]}) > 1),
        "first_search_suggestion": first, "llm_changed_first_suggestion": action["move"] != first,
        "search_seconds": sum(t["search_seconds"] for t in session["analyses"]),
        "evaluation": f"LLM final selection informed by {EVALUATION}; no numeric LLM score",
        "score_scope": "Evidence for selected move only; differing depths and upper bounds are not exact rankings"}
