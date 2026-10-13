"""Stages 2-4: validate raw files and build the point-in-time event table.

Reads only data/raw (never modifies it) and writes:
    data/interim/storage_clean.csv, prices_clean.csv
    data/processed/events.csv            (entry = 1st price after release)
    data/processed/events_lag2.csv       (entry = 2nd price after release; robustness)
    data/processed/prices_valid.csv, prices_all.csv
    data/processed/manifest.json         (hashes of the raw inputs used)
    reports/tables/data_validation.md, release_alignment_table.csv, missing_prices.csv

Usage:  python scripts/build_dataset.py
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from natgas_quant import data_download as DD  # noqa: E402
from natgas_quant import data_validation as DV  # noqa: E402
from natgas_quant import release_calendar as RC  # noqa: E402
from natgas_quant.config import ensure_dirs, load_config, resolve_path  # noqa: E402
from natgas_quant.dataset import alignment_table, build_event_table  # noqa: E402


def _latest_archive(raw: Path, name: str) -> Path | None:
    arch = raw / "archive"
    if not arch.exists():
        return None
    cands = sorted(arch.glob(f"*/{name}"))
    return cands[-1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = resolve_path(cfg["paths"]["raw"])
    interim = resolve_path(cfg["paths"]["interim"])
    proc = resolve_path(cfg["paths"]["processed"])
    tables = resolve_path(cfg["paths"]["tables"])
    src = cfg["sources"]

    storage_file = raw / "eia_storage_weekly.json"
    fr = src["fred_henry_hub"]
    price_file = raw / (f"fred_{fr['series_id']}.json" if fr["method"] == "api" else f"fred_{fr['series_id']}.csv")
    for f in (storage_file, price_file):
        if not f.exists():
            logging.error("Missing %s. Run: python scripts/download_data.py", f)
            return 1

    # ---------------- Parse + validate ----------------
    st_raw = DV.parse_eia_storage_json(storage_file)
    st, st_rep = DV.validate_storage(st_raw, src["eia_storage"]["series_id"],
                                     src["eia_storage"]["expected_units_contains"])
    px_raw = DV.parse_fred_api_json(price_file) if fr["method"] == "api" else DV.parse_fred_csv(price_file, fr["series_id"])
    px, px_rep = DV.validate_prices(px_raw)

    # Revision check against the previous vintage, if one was archived
    prev = _latest_archive(raw, storage_file.name)
    if prev is not None:
        old, _ = DV.validate_storage(DV.parse_eia_storage_json(prev), src["eia_storage"]["series_id"])
        rev = DV.compare_vintages(old, st, "week_ending_date", "storage_level_bcf")
        st_rep.facts["revised_weeks_vs_previous_vintage"] = len(rev)
        rev.to_csv(tables / "storage_revisions_vs_previous_vintage.csv", index=False)
    else:
        st_rep.facts["revised_weeks_vs_previous_vintage"] = "no earlier vintage archived"

    st.to_csv(interim / "storage_clean.csv", index=False)
    px.to_csv(interim / "prices_clean.csv", index=False)
    DV.missing_price_table(px).to_csv(tables / "missing_prices.csv", index=False)
    (tables / "data_validation.md").write_text(st_rep.to_markdown() + "\n\n" + px_rep.to_markdown() + "\n",
                                               encoding="utf-8")
    (tables / "data_validation.json").write_text(json.dumps([st_rep.to_dict(), px_rep.to_dict()], indent=2,
                                                            default=str), encoding="utf-8")

    # ---------------- Build events ----------------
    verified = RC.load_verified_releases(resolve_path(cfg["release"]["verified_release_file"]))
    ev, p, pdates = build_event_table(st, px, verified, cfg)
    cfg2 = copy.deepcopy(cfg)
    cfg2["release"]["entry_lag_trading_days"] = 2
    ev2, _, _ = build_event_table(st, px, verified, cfg2)

    ev.to_csv(proc / "events.csv", index=False)
    ev2.to_csv(proc / "events_lag2.csv", index=False)
    pd.DataFrame({"date": pdates, "price": p}).to_csv(proc / "prices_valid.csv", index=False)
    px[["date", "price"]].to_csv(proc / "prices_all.csv", index=False)
    alignment_table(ev).to_csv(tables / "release_alignment_table.csv", index=False)

    manifest = {"inputs": {}}
    for f in (storage_file, price_file):
        m = DD.meta_path_for(f)
        manifest["inputs"][f.name] = {
            "sha256": DD.sha256_of(f),
            "metadata": json.loads(m.read_text()) if m.exists() else "no sidecar (file not created by download script)",
        }
    manifest["built_at_utc"] = DD.utc_now_iso()
    (proc / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # ---------------- Manually inspectable sample weeks ----------------
    show = ["storage_week_ending_date", "report_release_date", "release_date_source", "price_before_release_date",
            "first_eligible_price_date", "storage_change_bcf", "z", "price_before_release", "price_after_release",
            "fwd_ret_1d"]
    ok = ev[ev["alignment_ok"] & ev["z"].notna()]
    normal = ok[ok["release_date_source"] == "rule_nominal"].head(3)
    holiday = ev[ev["release_date_source"] != "rule_nominal"].tail(6)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== Sample ordinary weeks (check: entry is the first price AFTER the release date) ===")
    print(normal[show].to_string(index=False))
    print("\n=== Sample holiday / verified / uncertain weeks ===")
    print(holiday[show].to_string(index=False))
    print(f"\nEvents: {len(ev)} weeks, {int(ev['alignment_ok'].sum())} aligned, "
          f"{int(ev['z'].notna().sum())} with anomaly z. Outputs in {proc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
