"""Stage 11: educational backtest (economic-signal simulation on spot prices).

    python scripts/run_backtest.py --phase development
        Chooses the threshold k on the VALIDATION period only, reports train
        and validation results, and freezes k.

    python scripts/run_backtest.py --phase final
        Runs the frozen rule once on the untouched test period, with
        no-trade, always-long and random-signal benchmarks.

Costs are set in configs/research_config.yaml (backtest.one_way_cost_bps,
backtest.slippage_bps) and recorded in the frozen specification.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from natgas_quant import analysis, reporting  # noqa: E402
from natgas_quant.config import ensure_dirs, load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["development", "final"], required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.phase == "development":
        res = analysis.run_backtest_development(cfg)
        print(f"k chosen on validation: {res['k']} (frozen spec {res['spec_hash']})")
        print("Next: python scripts/run_analysis.py --phase final")
    else:
        res = analysis.run_backtest_final(cfg)
        print(json.dumps(res, indent=2, default=str))
    print(f"Report written to {reporting.write_report(cfg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
