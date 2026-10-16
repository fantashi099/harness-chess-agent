"""Tool definitions exposed to the model, in the OpenAI tool-calling format."""

SELECT_AND_PLAY_TOOL = {
    "type": "function",
    "function": {
        "name": "select_and_play",
        "description": (
            "Run the selected chess decision mode on the latest observed position "
            "and commit exactly its selected White move. Black replies automatically. "
            "The server checks the position and prevents duplicate commits. "
            "Call once per turn until game_over; no move or code arguments."
        ),
        "strict": True,
        "parameters": {
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
        },
    },
}

EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": (
            "Run a bash command and return its stdout, stderr, and exit code. "
            "A non-zero exit code is reported, not raised.\n"
            "\n"
            "Every command runs in a new subshell, so a `cd` or an export does not "
            "carry over to the next command. Use the `cwd` and `env` arguments "
            "instead. Files you write do persist.\n"
            "\n"
            "Commands are non-interactive and cannot prompt for input, so pass "
            "flags like `-y` where a command would otherwise ask for confirmation. "
            "Prefer commands that produce little output; when reading a file, use "
            "`head`, `tail`, or `sed -n '10,20p'` rather than printing all of it.\n"
            "\n"
            "Useful patterns:\n"
            "- Create a file: `cat <<'EOF' > newfile.py` ... `EOF`\n"
            "- Edit in place: `sed -i 's/old/new/g' filename.py` (drop the trailing "
            "`g` to replace only the first match; restrict to a line range with "
            "`sed -i '1,10s/old/new/g'`)\n"
            "- View numbered lines: `nl -ba filename.py | sed -n '10,20p'`"
        ),
        # The nested env object intentionally accepts arbitrary variable names,
        # which is incompatible with strict schemas on some providers.
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "anyOf": [
                        {
                            "type": "string",
                            "description": 'A shell command line, e.g. "ls -la | head".',
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                'The command as an argv list, e.g. ["ls", "-la"]. '
                                "Use this with shell=false when arguments contain "
                                "characters the shell would interpret."
                            ),
                        },
                    ],
                    "description": "The command to run.",
                },
                "shell": {
                    "type": ["boolean", "null"],
                    "description": (
                        "Whether to run the command through a shell, which enables "
                        "pipes, redirection, and globbing. Defaults to true. Set to "
                        "false when passing an argv list."
                    ),
                },
                "cwd": {
                    "type": ["string", "null"],
                    "description": (
                        "Absolute path to run the command in. Defaults to the "
                        "sandbox's current working directory."
                    ),
                },
                "timeout": {
                    "type": ["number", "null"],
                    "description": (
                        "Seconds to allow the command to run before killing it. "
                        "Defaults to no timeout."
                    ),
                },
                "env": {
                    "type": ["object", "null"],
                    "additionalProperties": {"type": "string"},
                    "description": "Extra environment variables to set for this command.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

SEND_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": ("Send a message to the user."),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": ("Content of the message"),
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}

INVOKE_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "invoke_skill",
        "description": (
            "Load a skill and return its instructions. A skill is a short guide "
            "for one kind of work, written ahead of time.\n"
            "\n"
            "Call this before starting work a skill covers, and follow what it "
            "says in place of your default approach."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": (
                        "The skill's directory name, for example `hello-skill`."
                    ),
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
}

PLAY_MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "play_move",
        "description": (
            "Play one chess move as White on the live game board.\n"
            "\n"
            "Moves use UCI notation: two squares, from-square then to-square, "
            "with an optional promotion piece letter appended. Examples: "
            "`e2e4`, `g1f3`, `e7e8q` (promote to queen)."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": (
                        "The move in UCI notation (for example `e2e4` or `e7e8q`)."
                    ),
                },
            },
            "required": ["move"],
            "additionalProperties": False,
        },
    },
}

# TODO(3.3): Define the `simulate_move` tool, like the `play_move` tool.
SIMULATE_MOVE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "simulate_move",
        "description": (
            "Inspect a chess position or simulate one move WITHOUT changing "
            "the live game.\n"
            "\n"
            "Pass a complete six-field FEN (piece placement, active color, "
            "castling, en passant, halfmove, fullmove). With FEN alone, "
            "returns that position and its legal moves. With FEN plus a move "
            "in UCI notation (for example `e2e4` or `e7e8q`), returns the "
            "position after exactly that one ply, for either side. Use it to "
            "plan ahead before calling play_move."
        ),
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "fen": {
                    "type": "string",
                    "description": "A complete six-field FEN string.",
                },
                "move": {
                    "type": "string",
                    "description": (
                        "Optional. One move in UCI notation, applied to the "
                        "FEN position."
                    ),
                },
            },
            "required": ["fen"],
            "additionalProperties": False,
        },
    },
}

# TODO()
RUN_PYTHON_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "Run Python code in the game sandbox and read what it prints.\n"
            "\n"
            "The snippet can call two ordinary synchronous Python functions: "
            "simulate_move(fen, move=None) returns a dict with fen, squares, "
            "turn, legal_moves, in_check, game_over, winner, result; use those "
            "fields directly instead of parsing FEN or detecting attacks. play_move(move) "
            "commits one move to the live game and returns the new state. "
            "Use print() to report findings; this tool returns the captured "
            "stdout/stderr. Commit at most one play_move per snippet."
        ),
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}
