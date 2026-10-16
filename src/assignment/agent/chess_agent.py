"""The Part 3 chess agent: reuse the shared loop with one new domain tool."""

from __future__ import annotations

import json
from typing import Any

import httpx

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
)
from assignment.agent.chess_tools import (
    _game_state,
    _invoke_skill,
    _play_move,
    CHESS_PORT,
    _run_python,
    _simulate_move,
    _select_and_play,
)
from assignment.agent.chess_policy import position_id
from assignment.agent.tools import (
    INVOKE_SKILL_TOOL,
    PLAY_MOVE_TOOL,
    RUN_PYTHON_TOOL,
    SIMULATE_MOVE_TOOL,
    SELECT_AND_PLAY_TOOL,
)
from assignment.prompts import (
    CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE,
    CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE,
    PROGRAMMATIC_CHESS_PROMPT,
)
from assignment.env import Environment


def format_chess_state(
    state: dict[str, Any], *, include_legal_moves: bool = True
) -> str:
    """Turn chess API JSON into a compact observation an LLM can act on."""

    squares = state.get("squares", {})
    board_lines = ["    a b c d e f g h"]
    for rank in range(8, 0, -1):
        pieces = [squares.get(f"{file}{rank}", ".") for file in "abcdefgh"]
        board_lines.append(f"{rank} | {' '.join(pieces)} | {rank}")
    board_lines.append("    a b c d e f g h")

    recent_history = state.get("history", [])[-8:]
    history = (
        " ".join(
            f"{item.get('ply', '?')}:{item.get('san', '?')}" for item in recent_history
        )
        or "(none)"
    )
    legal_moves = " ".join(state.get("legal_moves", [])) or "(none)"
    board_text = "\n".join(board_lines)

    observation = (
        "<chess_state>\n"
        f"status: {state.get('status', 'unknown')}\n"
        f"turn: {state.get('turn', 'unknown')}\n"
        f"in_check: {state.get('in_check', False)}\n"
        f"game_over: {state.get('game_over', False)}\n"
        f"fen: {state.get('fen', '')}\n"
        f"human_move: {state.get('human_move') or '(none)'}\n"
        f"engine_move: {state.get('engine_move') or '(none)'}\n"
        "board:\n"
        f"{board_text}\n"
        f"recent_history: {history}\n"
    )
    if include_legal_moves:
        observation += f"legal_moves: {legal_moves}\n"
    return observation + "</chess_state>"


class ChessAgent(Agent):
    """An agent that plays White against the server's deterministic Black bot."""

    reasoning_effort = "low"
    max_completion_tokens = 8192

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 200,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        http_client: Any | None = None,
        reset_game: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = None,
        compaction_max_tokens: int = None,
        programmatic_tools: bool = False,
        python_sandbox_port: int = CHESS_PORT,
        include_legal_moves: bool = True,
        chess_policy: str | None = None,
    ):
        if chess_policy not in {None, "search", "harness"}:
            raise ValueError("Unsupported chess policy")
        if chess_policy and programmatic_tools:
            raise ValueError("Chess policy and arbitrary programmatic tools are separate modes.")
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )

        # TODO(Part 3): Register the play_move tool schema from tools.py.
        self.chess_policy = chess_policy
        self.planner_records = []
        self.chess_decisions: list[dict[str, Any]] = []
        self.tools.append(SELECT_AND_PLAY_TOOL if chess_policy else PLAY_MOVE_TOOL)
        if chess_policy=="search":
            self.tools[-1]=dict(SELECT_AND_PLAY_TOOL,function=dict(SELECT_AND_PLAY_TOOL["function"],
                description="Run bounded iterative alpha-beta search and commit one White move. Black replies automatically. Call with {} once per turn until game_over."))
        if chess_policy == "harness":
            self.tools[-1] = dict(SELECT_AND_PLAY_TOOL, function=dict(SELECT_AND_PLAY_TOOL["function"],
                description="Run the LLM planner/evaluator harness with bounded search tools; the inner LLM explicitly chooses one move. Call with {} once per White turn."))
        if chess_policy:
            self.max_completion_tokens = 512
        self.programmatic_tools = programmatic_tools

        if programmatic_tools:
            self.tools.append(RUN_PYTHON_TOOL)
            self.tools.append(SIMULATE_MOVE_TOOL)

        # run_python always executes in the sandbox, on the port the chess
        # server is listening on there.
        self.python_sandbox_port = python_sandbox_port
        self.include_legal_moves = include_legal_moves

        if http_client is None:
            server_url = getattr(environment, "server_url", "")
            if not server_url:
                raise ValueError(
                    "ChessAgent needs an environment with server_url or an http_client."
                )
            http_client = httpx.Client(base_url=server_url, timeout=20)
        self.chess_client = http_client

        initial_state = _game_state(self.chess_client, reset=reset_game)
        self.last_state = initial_state
        self.finished = bool(initial_state.get("game_over"))
        prompt_template = (
            CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE
            if include_legal_moves
            else CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE
        )
        self.system_prompt = prompt_template.render()
        if chess_policy=="search":
            self.system_prompt=("Play White. Call select_and_play with {} once per turn until game_over. "
                "The server performs bounded iterative alpha-beta search and commits one legal move. "
                "It retains the last completed iteration at budget cutoff. Scores are finite-horizon estimates. "
                "You cannot override the move, submit code, or execute a PV as multiple actions.")
        if chess_policy == "harness":
            self.system_prompt = ("Play White. Call select_and_play with {} once per turn until game_over. "
                "The inner LLM harness plans analysis, evaluates feedback, revises if needed and chooses the move. "
                "Code executes read-only bounded search and commits the explicit LLM choice.")
        if programmatic_tools:
            self.system_prompt += "\n\n" + PROGRAMMATIC_CHESS_PROMPT
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its instructions, and follow them in place "
                f"of your default approach.\n\n<skills>\n{catalog}\n</skills>\n"
            )
        if chess_policy:
            opening_instruction = "Call select_and_play with {} each turn."
        elif programmatic_tools and self.skills:
            opening_instruction = (
                "First invoke the relevant skill. Then use run_python to follow it: "
                "search with simulate_move and commit exactly one move with play_move."
            )
        elif programmatic_tools:
            opening_instruction = (
                "Use run_python to search with simulate_move and commit exactly one "
                "move with play_move."
            )
        elif include_legal_moves:
            opening_instruction = "Choose one move from legal_moves and call play_move."
        else:
            opening_instruction = (
                "Infer a legal move from the board and FEN, then call play_move."
            )
        self.task_prompt = (
            f"Play this game as White. {opening_instruction}\n\n"
            f"{self.format_state(initial_state)}"
        )

    def format_state(self, state: dict[str, Any]) -> str:
        """Format a state according to this run's observation condition."""

        return format_chess_state(state, include_legal_moves=self.include_legal_moves)

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute model-generated ``play_move`` calls against the chess API."""

        # TODO(Part 3.1):
        # 1. Dispatch on the function name, and ignore a tool this agent did
        #    not register.
        # 2. Hand the raw arguments to the matching chess_tools helper, with
        #    self.chess_client as its first argument. Each helper takes the
        #    client explicitly so the same code can run inside the sandbox.
        # 3. Format a played move with self.format_state, then update
        #    last_state and finished.
        # 4. Link every observation to its call with tool_call_id.
        # 5. Turn malformed, unknown, rejected, or extra parallel calls into
        #    recoverable <chess_error> observations instead of crashing.
        def format_error_message(error):
            return f"<chess_error>{error}</chess_error>"
        
        messages = []
        position_changing_call_seen = False
        handlers = (
            {"select_and_play": lambda args: _select_and_play(
                self.chess_client, args, position_id(self.last_state))}
            if self.chess_policy else
            {"play_move": lambda args: _play_move(self.chess_client, args)}
        )
        if self.chess_policy == "harness":
            from assignment.agent.chess_harness import harness_select_and_play
            handlers["select_and_play"] = lambda args: harness_select_and_play(self, args)
        if self.skills:
            handlers["invoke_skill"] = lambda args: _invoke_skill(self.skills, args)
        if self.programmatic_tools:
            handlers["simulate_move"] = lambda args: _simulate_move(self.chess_client, args)
            handlers["run_python"] = lambda args: _run_python(self.env, self.python_sandbox_port, args)

        for call in tool_calls:
            tool_name = call["function"]["name"]
            tool_call_id = call["id"]

            message = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
            }
            handler = handlers.get(tool_name)
            if handler is None:
                message["content"] = format_error_message(f"unknown tool '{tool_name}'")
                messages.append(message)
                continue

            args = call["function"]["arguments"]
            if tool_name in {"play_move", "run_python", "select_and_play"}:
                if position_changing_call_seen:
                    message["content"] = format_error_message(
                        "only one position-changing tool per response; position may "
                        "already have changed"
                    )
                    messages.append(message)
                    continue
                position_changing_call_seen = True
            result = handler(args)
            if tool_name == "select_and_play":
                if result.startswith("<chess_error>"):
                    # A lost response may follow a successful commit. Refresh
                    # rather than choosing from the previous root again.
                    try:
                        current_state = _game_state(self.chess_client)
                    except Exception as exc:
                        result += "\n" + format_error_message(f"State refresh failed: {exc}")
                    else:
                        self.last_state = current_state
                        self.finished = bool(current_state.get("game_over"))
                        result += "\n" + self.format_state(current_state)
                else:
                    payload = json.loads(result)
                    self.last_state = payload["state"]
                    self.finished = bool(self.last_state.get("game_over"))
                    decision = payload.get("decision")
                    summary = None
                    if decision is not None:
                        if decision["chosen_move"] != decision["committed_move"]:
                            raise RuntimeError("Server committed a move different from its policy.")
                        if not any(d["position_id"] == decision["position_id"]
                                   for d in self.chess_decisions):
                            self.chess_decisions.append(decision)
                        summary = {
                            key: decision.get(key) for key in (
                                "policy_version", "position_id", "root_fen", "chosen_move",
                                "primary_score", "secondary", "pv", "completed_depth",
                                "simulator_calls", "reason", "engine_move",
                            )
                        }
                        if self.chess_policy=="search":
                            summary.update(nodes_used=decision["nodes_used"],cutoff_reached=decision["cutoff_reached"],
                                           alpha_beta_prunes=decision["alpha_beta_prunes"],evaluation=decision["evaluation"])
                        elif self.chess_policy == "harness":
                            summary.update(nodes_used=decision["nodes_used"], cutoff_reached=decision["cutoff_reached"],
                                decision_authority="llm", analyses=len(decision["analysis_trace"]),
                                llm_changed_first_suggestion=decision["llm_changed_first_suggestion"])
                        summary["primary_tie_count"] = len(decision["primary_ties"])
                    result = json.dumps({"chess_decision": summary,
                                         "replayed": payload.get("replayed", False)})
                    result += "\n" + self.format_state(self.last_state)
            if tool_name == "play_move" and not result.startswith("<chess_error>"):
                current_state = json.loads(result)
                self.last_state = current_state
                self.finished = bool(current_state.get("game_over"))
                result = self.format_state(current_state)
            if tool_name == "run_python" and not result.startswith("<chess_error>"):
                try:
                    current_state = _game_state(self.chess_client)
                except Exception as exc:
                    result += "\n" + format_error_message(
                        "Python finished, but refreshing the live board failed after "
                        f"retries: {exc}. Call run_python with empty code to retry "
                        "the refresh before choosing another move."
                    )
                else:
                    self.last_state = current_state
                    self.finished = bool(current_state.get("game_over"))
                    result += "\n" + self.format_state(current_state)
            message["content"] = result
            messages.append(message)

        return messages

        # TODO(Part 3.3-4): add cases for simulate_move and run_python, with
        # linked observations and recoverable errors, just like the old tool.
