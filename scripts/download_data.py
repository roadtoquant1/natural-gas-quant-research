"""Stage 2: download raw data from EIA and FRED (cached; use --force to refresh).

Usage
-----
    python scripts/download_data.py            # use cache where present
    python scripts/download_data.py --force    # refresh, archiving the old vintage
    python scripts/download_data.py --include-optional   # also EIA spot/futures prices

Requires EIA_API_KEY in the environment or in a local .env file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from natgas_quant import data_download as DD  # noqa: E402
from natgas_quant.config import ensure_dirs, load_config, resolve_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--include-optional", action="store_true", help="also download EIA spot/futures prices")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw = resolve_path(cfg["paths"]["raw"])
    src = cfg["sources"]
    http = DD.HttpSettings(**src["http"])

    # ---------------- EIA storage ----------------
    es = src["eia_storage"]
    key = DD.get_api_key("EIA_API_KEY")
    meta = DD.fetch_eia_metadata(es["base_url"], es["route"], key, raw, http, facet_ids=("series",))
    route_resp = meta["route"]["response"]
    logging.info("EIA route '%s': %s", es["route"], route_resp.get("name") or route_resp.get("description"))
    logging.info("  facets available: %s", [f.get("id") for f in route_resp.get("facets", [])])
    logging.info("  frequencies: %s", [f.get("id") for f in route_resp.get("frequency", [])])
    entry = DD.find_facet_value(meta["facets"]["series"], es["series_id"])
    logging.info("Verified series %s: %s", es["series_id"], entry.get("name") or entry)
    DD.download_eia_series(
        es["base_url"], es["route"], es["series_id"], es["frequency"], es["start"], key,
        raw / "eia_storage_weekly.json", http, page_length=es["page_length"], force=args.force,
    )

    # ---------------- FRED Henry Hub ----------------
    fr = src["fred_henry_hub"]
    if fr["method"] == "api":
        DD.download_fred_api(fr["api_url"], fr["series_id"], DD.get_api_key("FRED_API_KEY"),
                             raw / f"fred_{fr['series_id']}.json", http, fr["units"], force=args.force)
    else:
        DD.download_fred_csv(fr["csv_url"], fr["series_id"], raw / f"fred_{fr['series_id']}.csv",
                             http, fr["units"], force=args.force)

    # ---------------- Optional EIA prices ----------------
    op = src["eia_prices"]
    if args.include_optional or op.get("enabled"):
        pm = DD.fetch_eia_metadata(es["base_url"], op["route"], key, raw, http, facet_ids=("series",))
        for sid in (op["spot_series_id"], op["futures_series_id"]):
            try:
                DD.find_facet_value(pm["facets"]["series"], sid)
            except DD.DownloadError as exc:
                logging.warning("Optional series unavailable, skipped: %s", exc)
                continue
            DD.download_eia_series(es["base_url"], op["route"], sid, op["frequency"], "1997-01-01", key,
                                   raw / f"eia_price_{sid}.json", http, force=args.force)

    logging.info("Done. Raw files and .meta.json sidecars are in %s", raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
