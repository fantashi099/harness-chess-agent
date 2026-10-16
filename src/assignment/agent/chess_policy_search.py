"""Iterative alpha-beta with bounded quiescence; owns no opponent engine."""
from __future__ import annotations

import chess

if __package__:
    from assignment.agent.chess_policy import VALUES, CENTERS, secondary, position_id, INITIAL_FEN
else:
    from chess_policy import VALUES, CENTERS, secondary, position_id, INITIAL_FEN

POLICY_VERSION = "select-move-v5-search-1"
SEARCH_VERSION = "iterative-alpha-beta-q6-2"
QDEPTH = 6
MATE = 100_000
INF = 1_000_000


def identity(board, game_id):
    return position_id({"fen":board.fen(), "history":[m.uci() for m in board.move_stack], "policy_game_id":game_id})


def static_score(board):
    """Exact unchanged v2 NONTERMINAL White score, without a state dict allocation."""
    value=sum(VALUES[p.symbol().upper()]*(1 if p.color else -1) for p in board.piece_map().values())
    for name in CENTERS:
        piece=board.piece_at(chess.parse_square(name))
        if piece:value+=12 if piece.color else -12
    if board.is_check():value+=25 if not board.turn else -25
    return value


def positional_score(board):
    """Small v6 heuristic; king danger and pawn weaknesses supplement material."""
    value = static_score(board)
    for color in (chess.WHITE, chess.BLACK):
        own = board.occupied_co[color]
        pawns = board.pieces(chess.PAWN, color)
        enemy_pawns = board.pieces(chess.PAWN, not color)
        files = [sum(chess.square_file(s) == f for s in pawns) for f in range(8)]
        score = -12 * sum(max(0, count-1) for count in files)
        for square in pawns:
            f, r = chess.square_file(square), chess.square_rank(square)
            if not any(files[n] for n in (f-1, f+1) if 0 <= n < 8):
                score -= 10
            if not any(abs(chess.square_file(s)-f) <= 1 and
                       (chess.square_rank(s) > r if color else chess.square_rank(s) < r)
                       for s in enemy_pawns):
                score += 8 * (r if color else 7-r)
        for piece in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            for square in board.pieces(piece, color):
                score += 2 * chess.popcount(board.attacks_mask(square) & ~own)
                if piece == chess.ROOK and not files[chess.square_file(square)]:
                    score += 10
        king = board.king(color)
        if king is not None and (board.pieces(chess.QUEEN, not color) or board.pieces(chess.ROOK, not color)):
            ring = board.attacks_mask(king) | chess.BB_SQUARES[king]
            score -= 8 * sum(len(board.attackers(not color, s)) for s in chess.scan_forward(ring))
            rank = chess.square_rank(king)
            if (rank if color else 7-rank) <= 2:
                for f in range(max(0, chess.square_file(king)-1), min(8, chess.square_file(king)+2)):
                    if not any(chess.square_file(s) == f and 1 <=
                               ((chess.square_rank(s)-rank) if color else (rank-chess.square_rank(s))) <= 2
                               for s in pawns):
                        score -= 15
                    if not files[f]:
                        score -= 15
            else:
                score -= 30
        value += score if color else -score
    return value


def undeveloped_minors(board, color):
    rank = 0 if color else 7
    return [chess.square(f, rank) for f, kind in ((1, chess.KNIGHT), (2, chess.BISHOP),
                                                (5, chess.BISHOP), (6, chess.KNIGHT))
            if board.piece_at(chess.square(f, rank)) == chess.Piece(kind, color)]


def development_score(board):
    """Phase-weighted development and castling supplement the original v6 evaluator."""
    value = positional_score(board)
    phase = min(24, sum(weight * len(board.pieces(kind, color))
                       for color in (chess.WHITE, chess.BLACK)
                       for kind, weight in ((chess.KNIGHT, 1), (chess.BISHOP, 1),
                                            (chess.ROOK, 2), (chess.QUEEN, 4))))
    for color in (chess.WHITE, chess.BLACK):
        home = 0 if color else 7
        minors = undeveloped_minors(board, color)
        score = -32 * len(minors)
        for square in board.pieces(chess.QUEEN, color):
            if square != chess.square(3, home):
                score -= 12 * len(minors)
        for square in board.pieces(chess.KNIGHT, color):
            f, r = chess.square_file(square), chess.square_rank(square)
            score += 6 * min(f, 7-f, r, 7-r)
        king = board.king(color)
        if king == chess.square(4, home):
            if not board.has_castling_rights(color):
                score -= 45
        elif king in (chess.square(2, home), chess.square(6, home)):
            score += 35
        elif king is not None:
            score -= 40
        # shortcut: home squares approximate development without opening history; revisit with broader game evidence.
        value += (1 if color else -1) * (score * phase // 24)
    return value


class BudgetExhausted(Exception):
    pass


def select_move(snapshot, game_id, max_nodes=6000, max_depth=4, candidate_moves=None,
                evaluation="unchanged-v2-static"):
    if type(max_nodes) is not int or max_nodes<1 or type(max_depth) is not int or not 1<=max_depth<=12:
        raise ValueError("Positive node budget and depth 1..12 required")
    if not isinstance(snapshot,chess.Board) or not snapshot.is_valid() or not isinstance(game_id,str) or not game_id:
        raise ValueError("A valid board with game identity is required")
    evaluators = {"unchanged-v2-static": static_score,
                  "king-safety-mobility-pawns-v1": positional_score,
                  "development-king-safety-v2": development_score}
    if evaluation not in evaluators:
        raise ValueError("Unknown evaluator")
    evaluate = evaluators[evaluation]
    board=snapshot.copy(stack=True)
    if candidate_moves is not None and (not isinstance(candidate_moves, list) or not candidate_moves
            or any(not isinstance(m, str) or m not in {move.uci() for move in board.legal_moves} for m in candidate_moves)
            or len(set(candidate_moves)) != len(candidate_moves)):
        raise ValueError("Candidate scope must contain unique legal UCI moves")
    root={"fen":board.fen(),"squares":{chess.square_name(s):p.symbol() for s,p in board.piece_map().items()}}
    recent=[m.uci() for i,m in enumerate(board.move_stack) if (board.root().turn if i%2==0 else not board.root().turn)==chess.WHITE][-4:]
    nodes,prunes,qnodes,qhorizons=1,0,0,0
    iterations=[]
    ordering={}  # Move hints only: no history-unsafe transposition score cache.
    common={"policy_version":POLICY_VERSION,"search_version":SEARCH_VERSION,"root_fen":board.fen(),
        "executor_position_id":identity(board,game_id),"recent_white_moves":recent,
        "repetition_history_available":True,"simulator_backend":"immutable_board_stack",
        "search_config":{"max_depth":max_depth},"max_nodes":max_nodes,"evaluation":evaluation,
        "terminal_scoring":"Mate distance breaks winning/losing terminal ties; native draw claims preserved",
        "primary_ties":[],"top_candidates":[],"secondary":{},"completed_depth":0,"search_performed":False,
        "score_scope":"Bounded main search plus quiescence estimate; depth boundaries are approximate, not game proofs"}

    def finish(**kwargs):
        return dict(common,nodes_used=nodes,simulator_calls=nodes,alpha_beta_prunes=prunes,
                    quiescence={"max_depth":QDEPTH,"nodes":qnodes,"horizon_leaves":qhorizons,
                                "scope":"Captures, promotions and all legal check evasions; static boundary at the depth limit"},
                    iterations=iterations,**kwargs)

    outcome=board.outcome(claim_draw=True)
    if outcome:
        value=0 if outcome.winner is None else MATE if outcome.winner else -MATE
        return finish(chosen_move=None,primary_score=value,pv=[],cutoff_reached=False,reason="terminal_position")
    if not board.turn:raise ValueError("v5 selects only White moves")
    if board.fen()==INITIAL_FEN and candidate_moves is None and evaluation == "unchanged-v2-static":
        return finish(chosen_move="e2e4",primary_score=None,pv=["e2e4"],cutoff_reached=False,reason="opening_preference")

    def tick():
        nonlocal nodes
        if nodes>=max_nodes:raise BudgetExhausted
        nodes+=1

    def ordered(preferred=None, root_moves=False):
        hint=preferred or ordering.get(board.fen())
        def key(move):
            text=move.uci()
            if root_moves:
                _,features=secondary(root,text,recent)
                return (int(text==hint),*features)
            victim=board.piece_at(move.to_square)
            captured=VALUES[victim.symbol().upper()] if victim else 100 if board.is_en_passant(move) else 0
            attacker=board.piece_at(move.from_square)
            tactical=10*captured-(VALUES[attacker.symbol().upper()] if captured else 0)
            return (int(text==hint),int(bool(move.promotion)),tactical,int(board.is_castling(move)))
        moves = board.legal_moves
        if root_moves and candidate_moves is not None:
            moves = [m for m in moves if m.uci() in candidate_moves]
        return sorted(sorted(moves,key=lambda m:m.uci()),key=key,reverse=True)

    def search(depth,alpha,beta,ply,qleft=QDEPTH):
        nonlocal prunes,qnodes,qhorizons
        tick()
        end=board.outcome(claim_draw=True)
        if end:
            white=0 if end.winner is None else MATE-ply if end.winner else -MATE+ply
            return (white if board.turn else -white),[]
        quiet=depth==0
        checked=board.is_check()
        best,line=-INF,[]
        if quiet:
            qnodes+=1
            white=evaluate(board)
            stand=white if board.turn else -white
            if qleft==0:
                # ponytail: a bounded tactical estimate; extend only if observed horizon errors persist.
                qhorizons+=int(checked or any(board.is_capture(m) or m.promotion for m in board.legal_moves))
                return stand,[]
            if not checked:
                best=stand
                if best>=beta:
                    prunes+=1
                    return best,[]
                alpha=max(alpha,best)
        key=board.fen()
        for move in ordered():
            if quiet and not checked and not (board.is_capture(move) or move.promotion):continue
            board.push(move)
            try:score,pv=search(0 if quiet else depth-1,-beta,-alpha,ply+1,qleft-1 if quiet else QDEPTH);score=-score
            finally:board.pop()
            if score>best:best,line=score,[move.uci()]+pv
            alpha=max(alpha,best)
            if alpha>=beta:
                prunes+=1
                break
        if line:ordering[key]=line[0]
        return best,line

    fallback=ordered(root_moves=True)[0].uci()
    best={"move":fallback,"primary_score":None,"pv":[fallback]}
    cutoff=False
    for depth in range(1,max_depth+1):
        started=nodes
        rows=[];alpha=-INF
        try:
            tick()
            for move in ordered(best["move"],True):
                prior_alpha=alpha
                board.push(move)
                try:score,pv=search(depth-1,-INF,-alpha,1);score=-score
                finally:board.pop()
                features,_=secondary(root,move.uci(),recent)
                rows.append({"move":move.uci(),"primary_score":score,"pv":[move.uci()]+pv,
                             "secondary":features,"bound":"exact" if score>prior_alpha else "upper"})
                alpha=max(alpha,score)
        except BudgetExhausted:
            iterations.append({"depth":depth,"completed":False,"nodes":nodes-started,"stop_reason":"max_nodes"})
            cutoff=True
            break
        # A fail-low candidate cannot displace the exact best score on a tie.
        best=max(rows,key=lambda r:r["primary_score"])
        common.update(completed_depth=depth,search_performed=True,secondary=best["secondary"],
                      top_candidates=sorted(rows,key=lambda r:-r["primary_score"])[:5])
        iterations.append({"depth":depth,"completed":True,"nodes":nodes-started,
                           "move":best["move"],"primary_score":best["primary_score"],"pv":best["pv"]})
        if best["primary_score"]>=MATE-depth:break
    return finish(chosen_move=best["move"],primary_score=best["primary_score"],pv=best["pv"],
                  cutoff_reached=cutoff,reason="last_completed_iteration" if common["completed_depth"] else "legal_fallback_unsearched")
