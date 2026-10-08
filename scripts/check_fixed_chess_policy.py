"""Run the four fixed-policy regressions without an LLM or Modal."""

import argparse
import json
from pathlib import Path

from assignment.agent.chess_policy import POLICY_VERSION, select_move
from assignment.chess_server import simulate_position

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path,
                        default=REPO / "experiments/chess/positions/fixed_policy_regressions.json")
    parser.add_argument("--output", type=Path,
                        default=REPO / "artifacts/assignment2/fixed-policy-regressions.json")
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    records = []
    for fixture in fixtures["positions"]:
        decision = select_move(fixture["fen"], simulate_position)
        passed = (decision["chosen_move"] == fixture["v2_chosen_move"]
                  and decision["primary_score"] == fixture["v2_best_score"]
                  and decision["chosen_move"] != fixture["played_move"])
        records.append({"fixture": fixture, "passed": passed, "decision": decision})
    report = {"policy_version": POLICY_VERSION, "passed": sum(r["passed"] for r in records),
              "total": len(records), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Fixed v2 regressions: {report['passed']}/{report['total']} passed.")
    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
