"""Stages 5-10 and 13: statistical analysis and report.

    python scripts/run_analysis.py --phase development
        Train + validation data only. Audit, EDA, event study, inference,
        validation forecasts, robustness. Freezes the model specification.

    python scripts/run_analysis.py --phase final --note "first and only test evaluation"
        Requires frozen model AND backtest specifications. Evaluates the
        untouched test period once and logs the evaluation.

    python scripts/run_analysis.py --phase report
        Rebuilds reports/research_report.md from whatever tables exist.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from natgas_quant import analysis, reporting  # noqa: E402
from natgas_quant.config import ensure_dirs, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["development", "final", "report"], required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--note", default="", help="free-text note stored in the test-evaluation log")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.phase == "development":
        res = analysis.run_development(cfg)
        print(f"Development phase complete. Frozen model spec {res['spec_hash']}.")
        print("Next: python scripts/run_backtest.py --phase development")
    elif args.phase == "final":
        res = analysis.run_final(cfg, note=args.note)
        print("Final (test) evaluation complete:")
        for k, v in res.items():
            print(f"  {k}: {v}")
    path = reporting.write_report(cfg)
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
