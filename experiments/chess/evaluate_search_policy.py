"""Run the actual ChessAgent tool handler in Modal for the requested game count."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import monotonic

import chess

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from assignment.agent.chess_agent import ChessAgent
from assignment.chess_artifacts import save_fixed_run,search_sources,harness_sources
from assignment.chess_sandbox import ChessSandbox
from scripts.replay_v3_chess_game import verify

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n")


def run(output,games,max_nodes,depth,steps=200,policy="v5",model=None,stockfish_elo=1500):
    if (output/"experiment-config.json").exists():raise FileExistsError("Choose a fresh output directory")
    output.mkdir(parents=True,exist_ok=True)
    model = (model or os.environ.get("OPENAI_MODEL")) if policy == "v6" else "deterministic-v5"
    if not model:
        raise ValueError("v6 requires --model or OPENAI_MODEL")
    config={"policy":policy,"model":model,"max_nodes":max_nodes,"max_depth":depth,"max_games":games,"step_limit":steps,
        "stop_after_replay_validated_win":False,"opponent":{"mode":"stockfish","name":"Stockfish 17.1",
        "elo":stockfish_elo,"move_time":0.2,"threads":1,"hash_mib":64},"teacher_used":False,
        "action_driver":("ChessAgent.execute_tool_calls; real inner LLM planning/evaluation/commit, no outer LLM"
            if policy == "v6" else "ChessAgent.execute_tool_calls; deterministic controller, no outer LLM API calls")}
    write(output/"experiment-config.json",config)
    sources={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (harness_sources(REPO) if policy == "v6" else search_sources(REPO))}
    for relative in sources:
        dest=output/"source-snapshot"/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/relative,dest)
    write(output/"source-freeze.json",sources)
    summaries=[]
    for number in range(1,games+1):
        directory=output/f"game_{number}";directory.mkdir()
        started=monotonic();actions=[];stop_reason="incomplete";failure_reason=None
        print(f"Starting game {number}: {max_nodes} nodes, target depth {depth}",flush=True)
        with ChessSandbox(patch=REPO/"artifacts/fix.patch",fixed_chess_policy=policy,max_nodes=max_nodes,
                          search_depth=depth,opponent="stockfish",stockfish_elo=stockfish_elo,stockfish_time=0.2,
                          deployment_timeout=3600) as sandbox:
            print(f"Chess server: {sandbox.server_url}",flush=True)
            agent=ChessAgent(sandbox,model=model,fixed_chess_policy=policy,auto_stop_environment=False,step_limit=steps)
            agent.action_driver=config["action_driver"]
            assert agent.last_state["fen"]==chess.STARTING_FEN and not agent.last_state["history"]
            assert agent.last_state["opponent"]==config["opponent"]
            failures=0
            try:
                for step in range(1,steps+1):
                    if agent.finished:break
                    call={"id":f"turn-{step}","function":{"name":"select_and_play","arguments":"{}"}}
                    observation=agent.execute_tool_calls([call])
                    actions.append({"step":step,"tool_call":call,"observation":observation})
                    errors = [o["content"].split("</chess_error>", 1)[0] + "</chess_error>"
                              for o in observation if "<chess_error>" in o["content"]]
                    if errors:
                        failures+=1
                        print(json.dumps({"game": number, "step": step, "status": "action_failed",
                            "errors": errors, "consecutive_failures": failures, "fen": agent.last_state["fen"]}), flush=True)
                        if failures>=3 or any(reason in error for error in errors for reason in (
                                "LLM call allowance exhausted", "No commit-ready move and analysis budget exhausted")):
                            stop_reason="action_failed"
                            failure_reason=f"Game {number}, step {step}: {'; '.join(errors)}"
                            break
                        continue
                    failures=0
                    decision=agent.chess_decisions[-1] if agent.chess_decisions else {}
                    print(json.dumps({"game":number,"step":step,"move":decision.get("chosen_move"),
                        "depth":decision.get("completed_depth"),"nodes":decision.get("nodes_used"),
                        "black":decision.get("engine_move"),"game_over":agent.finished}),flush=True)
                if failure_reason is None:
                    stop_reason="terminal" if agent.finished else "step_limit"
                assert not agent.api_responses
                if policy == "v5":
                    assert not agent.planner_records
                else:
                    assert agent.planner_records and all(d["reason"] == "explicit_llm_commit" for d in agent.chess_decisions)
            except BaseException as exc:
                stop_reason=type(exc).__name__
                raise
            finally:
                save_fixed_run(agent,directory/"result.json",REPO/"artifacts/fix.patch",stop_reason)
                write(directory/"trajectory.json",{"action_driver":config["action_driver"],"steps":actions})
        assert all(hashlib.sha256((REPO/p).read_bytes()).hexdigest()==v for p,v in sources.items()),"Source changed during evaluation"
        report=verify(directory/"result.json",REPO/"artifacts/fix.patch")
        write(directory/"replay-report.json",report)
        result=json.loads((directory/"result.json").read_text())
        decisions=[json.loads(s) for s in (directory/"decisions.jsonl").read_text().splitlines()]
        summary={"game":number,"validated":report["validated"],"completed":report["completed"],
            "benchmark_outcome": ("win" if report["validated"] and report["completed"] and report["result"] == "1-0"
                else "draw" if report["validated"] and report["completed"] and report["result"] == "1/2-1/2" else "loss"),
            "result":report["result"],"plies":report["plies"],"white_decisions":len(decisions),
            "mean_completed_depth":sum(d["completed_depth"] for d in decisions)/len(decisions) if decisions else None,
            "nodes":sum(d["nodes_used"] for d in decisions),"seconds_including_replay":monotonic()-started,
            "artifact_errors":result["artifact_errors"], "stop_reason":stop_reason, "failure_reason":failure_reason}
        if policy == "v6":
            manifest = json.loads((directory/"manifest.json").read_text())
            summary.update(llm_responses=manifest["llm_response_count"], llm_token_usage=manifest["llm_token_usage"],
                changed_first_suggestion=manifest["changed_first_suggestion"], replanned_turns=manifest["replanned_turns"],
                verification_analyses=manifest["verification_analyses"], comparison_turns=manifest["comparison_turns"],
                repair_turns=manifest["repair_turns"],
                llm_choice_verified=report["all_commits_match_llm_response"],
                risk_accepted_turns=sum(bool(d.get("risk_acceptance")) for d in decisions),
                continuation_warnings=sum(bool(d.get("continuation_checks", {}).get("error")) for d in decisions))
        summaries.append(summary);write(output/"summary.json",summaries)
        print(json.dumps(summary),flush=True)
    return summaries


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--games",type=int,default=3)
    parser.add_argument("--max-nodes",type=int,default=6000)
    parser.add_argument("--depth",type=int,default=4)
    parser.add_argument("--steps",type=int,default=200)
    parser.add_argument("--policy",choices=["v5", "v6"],default="v5")
    parser.add_argument("--model",help="v6 planner/evaluator model; defaults to OPENAI_MODEL")
    strength=parser.add_mutually_exclusive_group()
    strength.add_argument("--stockfish-elo",type=int,default=1500,
                          help="Stockfish strength target within the engine's supported Elo range (default: 1500)")
    strength.add_argument("--stockfish-unlimited",action="store_true",
                          help="disable Elo strength limiting and use Skill Level 20; still 0.2 seconds per move")
    args=parser.parse_args(argv)
    if args.games<1 or args.steps<1:parser.error("Positive game and step limits required")
    run(args.output.resolve(),args.games,args.max_nodes,args.depth,args.steps,args.policy,args.model,
        stockfish_elo=None if args.stockfish_unlimited else args.stockfish_elo)


if __name__=="__main__":
    main()
