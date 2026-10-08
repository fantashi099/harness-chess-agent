from jinja2 import Template

CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE = Template(
    """You are playing White in a chess game against a deterministic Black bot.

Use the `play_move` tool for every move. Pass exactly one UCI move listed in the
latest `legal_moves` field, then inspect the returned board before choosing the
next move. Uppercase pieces are White; lowercase pieces are Black; `.` is an
empty square. The server applies Black's reply automatically, so never submit a
move for Black and never assume what Black played. Promotion moves include a
piece suffix, for example `e7e8q`.

Continue until the returned state says `game_over: true`. While the game is
active, make a tool call instead of merely describing a move in text."""
)
"""System prompt for the chess-playing agent."""

CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE = Template(
    """You are playing White in a chess game against a deterministic Black bot.

Use the `play_move` tool for every move. Infer a legal UCI move from the latest
board and FEN, then inspect the returned board before choosing the next move.
Uppercase pieces are White; lowercase pieces are Black; `.` is an empty square.
The server applies Black's reply automatically, so never submit a move for Black
and never assume what Black played. Promotion moves include a piece suffix, for
example `e7e8q`.

Continue until the returned state says `game_over: true`. While the game is
active, make a tool call instead of merely describing a move in text."""
)
"""Ablation prompt used when legal moves are omitted from observations."""

PROGRAMMATIC_CHESS_PROMPT = """Two additional tools are available: simulate_move and run_python.
You may use Python to explore hypothetical positions before choosing a move.
simulate_move(fen, move=None) returns a dict with fen, squares, turn, legal_moves,
in_check, and terminal status. Use those returned fields directly; do not write a
FEN parser or reimplement attack detection. Keep each Python snippet concise. Use
simulate_move to search for a good move, then call play_move exactly once as the
last action in the snippet.

Chess-quality rule: this rule takes priority over a skill's alphabetical
tie-break. When primary search scores tie, prefer castling, developing a knight or
bishop from its starting square, and advancing a central pawn, in that order.
Penalize early king, queen, and rook moves, returning a developed piece to its
starting square, and repeating recent moves. Use alphabetical order only if these
chess tie-breaks are also equal."""

FIXED_CHESS_PROMPT = """You are playing White against the deterministic Black bot.
Use select_and_play with {} exactly once per turn. This tool runs the fixed v2
policy in the sandbox, checks the current position, and commits its chosen move.
The returned observation includes the decision summary, Black's reply, and the
next live state. Continue until game_over: true.

You cannot submit a different move, write Python, or bypass the policy. After a
tool error, use the latest refreshed state and retry select_and_play. If several
tool calls are emitted together, only one position-changing call is accepted.
When skills are available, invoke select-move-v2 once before playing."""
