"""Download raw data from primary public sources and save untouched copies.

Design rules
------------
1. The bytes returned by the server are written to ``data/raw`` unchanged.
   Parsing happens later, in :mod:`natgas_quant.data_validation`.
2. Every raw file gets a ``.meta.json`` sidecar recording the URL, query
   (with any API key redacted), series id, units, frequency, download time,
   date range and row count.
3. If a raw file already exists we reuse it (a cache) unless ``force=True``.
   On a forced refresh the old file is moved to ``data/raw/archive/<stamp>/``
   rather than deleted, so earlier data *vintages* are kept and revisions can
   be measured later.
4. HTTP and API errors raise informative exceptions; nothing is silently
   replaced.

Vocabulary
----------
*API* (application programming interface): a web address that returns data in
a machine-readable form such as JSON.
*Facet*: an EIA filter dimension (e.g. ``series`` or ``duoarea``) whose
allowed values can be listed through a metadata endpoint.
*Vintage*: the version of a historical number as it was published on a given
date. Agencies revise data, so today's download is the *current vintage*.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import requests

log = logging.getLogger(__name__)

REDACTED = "***REDACTED***"


class DownloadError(RuntimeError):
    """Raised when a request fails after all retries or the payload is unusable."""


class APIResponseError(DownloadError):
    """Raised when a server answers with an explicit API error message."""


class _SessionLike(Protocol):
    def get(self, url: str, params: dict[str, Any] | None = ..., timeout: float = ...) -> Any: ...


@dataclass(frozen=True)
class HttpSettings:
    """Retry and timeout settings for HTTP requests."""

    timeout_seconds: float = 60.0
    max_retries: int = 4
    backoff_seconds: float = 2.0


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------
def get_api_key(name: str, required: bool = True) -> str | None:
    """Read an API key from the environment (optionally loaded from ``.env``).

    The key is never printed or written to disk by this module.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # python-dotenv is optional at runtime
        pass
    value = os.environ.get(name, "").strip()
    if not value or value.startswith("replace_with"):
        if required:
            raise DownloadError(
                f"Environment variable {name} is not set. Copy .env.example to .env "
                f"and add your key (see README, 'Environment variables')."
            )
        return None
    return value


def redact_params(params: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of query parameters with anything key-like redacted."""
    out: dict[str, Any] = {}
    for k, v in params.items():
        out[k] = REDACTED if "key" in k.lower() else v
    return out


# ---------------------------------------------------------------------------
# HTTP with retries
# ---------------------------------------------------------------------------
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def http_get(
    url: str,
    params: dict[str, Any] | None = None,
    settings: HttpSettings = HttpSettings(),
    session: _SessionLike | None = None,
    sleep: Any = time.sleep,
) -> requests.Response:
    """GET a URL with exponential-backoff retries on transient failures.

    Retries on connection errors and on HTTP 429/5xx. Any other non-200 status
    raises immediately, including the server's message, because retrying a
    malformed request cannot succeed.
    """
    sess = session if session is not None else requests.Session()
    last_exc: Exception | None = None
    for attempt in range(settings.max_retries + 1):
        try:
            resp = sess.get(url, params=params, timeout=settings.timeout_seconds)
        except requests.RequestException as exc:  # network-level failure
            last_exc = exc
            log.warning("Request error on attempt %d: %s", attempt + 1, type(exc).__name__)
        else:
            if resp.status_code == 200:
                return resp
            body = (resp.text or "")[:500]
            if resp.status_code not in _RETRYABLE_STATUS:
                raise APIResponseError(
                    f"HTTP {resp.status_code} from {url}: {body}"
                )
            last_exc = DownloadError(f"HTTP {resp.status_code}: {body}")
            log.warning("Retryable HTTP %d on attempt %d", resp.status_code, attempt + 1)
        if attempt < settings.max_retries:
            sleep(settings.backoff_seconds * (2**attempt))
    raise DownloadError(
        f"Giving up on {url} after {settings.max_retries + 1} attempts: {last_exc}"
    )


def check_eia_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Raise if an EIA v2 JSON payload contains an error; return its ``response``.

    EIA v2 errors look like ``{"error": "...", "code": 400}``. Non-fatal
    problems appear as a ``warning`` field, which we log loudly.
    """
    if "error" in payload:
        raise APIResponseError(f"EIA API error (code {payload.get('code')}): {payload['error']}")
    if "response" not in payload:
        raise APIResponseError(f"EIA payload has no 'response' key; keys={list(payload)}")
    response = payload["response"]
    for w in response.get("warnings", []) or []:
        log.warning("EIA API warning: %s", w)
    if payload.get("warning"):
        log.warning("EIA API warning: %s", payload["warning"])
    return response


# ---------------------------------------------------------------------------
# Raw-file bookkeeping
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_existing(path: Path, raw_dir: Path) -> Path | None:
    """Move an existing raw file (and its sidecar) into ``raw/archive/<stamp>/``."""
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest_dir = raw_dir / "archive" / stamp
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    shutil.move(str(path), dest)
    meta = meta_path_for(path)
    if meta.exists():
        shutil.move(str(meta), dest_dir / meta.name)
    log.info("Archived previous vintage to %s", dest_dir)
    return dest


def meta_path_for(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def write_metadata(path: Path, meta: dict[str, Any]) -> Path:
    meta = dict(meta)
    meta["sha256"] = sha256_of(path)
    meta["file"] = path.name
    mp = meta_path_for(path)
    mp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return mp


# ---------------------------------------------------------------------------
# EIA v2
# ---------------------------------------------------------------------------
def eia_get(
    base_url: str,
    path: str,
    params: dict[str, Any],
    settings: HttpSettings,
    session: _SessionLike | None = None,
) -> dict[str, Any]:
    """GET an EIA v2 endpoint and return the full JSON payload after error checks."""
    url = f"{base_url.rstrip('/')}/{path.strip('/')}"
    resp = http_get(url, params=params, settings=settings, session=session)
    try:
        payload = resp.json()
    except ValueError as exc:
        raise APIResponseError(f"EIA returned non-JSON content from {url}") from exc
    check_eia_payload(payload)
    return payload


def fetch_eia_metadata(
    base_url: str,
    route: str,
    api_key: str,
    raw_dir: Path,
    settings: HttpSettings,
    facet_ids: tuple[str, ...] = ("series",),
    session: _SessionLike | None = None,
) -> dict[str, Any]:
    """Save the route metadata and the listed facet values to ``raw_dir/eia_metadata``.

    Returns a dict ``{"route": payload, "facets": {facet_id: payload}}``.
    """
    out_dir = raw_dir / "eia_metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = route.replace("/", "_")
    route_payload = eia_get(base_url, route, {"api_key": api_key}, settings, session)
    rp = out_dir / f"{safe}_route.json"
    rp.write_text(json.dumps(route_payload, indent=2), encoding="utf-8")
    write_metadata(rp, {"url": f"{base_url}/{route}", "query": {"api_key": REDACTED},
                        "downloaded_at_utc": utc_now_iso()})
    facets: dict[str, Any] = {}
    for fid in facet_ids:
        fp_payload = eia_get(base_url, f"{route}/facet/{fid}", {"api_key": api_key}, settings, session)
        fp = out_dir / f"{safe}_facet_{fid}.json"
        fp.write_text(json.dumps(fp_payload, indent=2), encoding="utf-8")
        write_metadata(fp, {"url": f"{base_url}/{route}/facet/{fid}",
                            "query": {"api_key": REDACTED},
                            "downloaded_at_utc": utc_now_iso()})
        facets[fid] = fp_payload
    return {"route": route_payload, "facets": facets}


def find_facet_value(facet_payload: dict[str, Any], value_id: str) -> dict[str, Any]:
    """Return the facet entry whose id equals ``value_id`` or raise.

    This is how we *verify* that the series we intend to download exists,
    instead of guessing.
    """
    facets = facet_payload.get("response", {}).get("facets", [])
    for entry in facets:
        if str(entry.get("id")) == value_id:
            return entry
    sample = [e.get("id") for e in facets[:15]]
    raise DownloadError(
        f"Series '{value_id}' not found in EIA facet listing ({len(facets)} values; "
        f"first few: {sample}). Inspect data/raw/eia_metadata and update the config."
    )


def download_eia_series(
    base_url: str,
    route: str,
    series_id: str,
    frequency: str,
    start: str,
    api_key: str,
    out_path: Path,
    settings: HttpSettings,
    page_length: int = 5000,
    force: bool = False,
    session: _SessionLike | None = None,
) -> Path:
    """Download every row of one EIA v2 series, paging through results.

    The untouched list of page payloads is written as JSON to ``out_path``.
    """
    raw_dir = out_path.parent
    if out_path.exists() and not force:
        log.info("Using cached %s (pass --force to refresh)", out_path.name)
        return out_path
    archive_existing(out_path, raw_dir)
    params: dict[str, Any] = {
        "api_key": api_key,
        "frequency": frequency,
        "data[0]": "value",
        "facets[series][]": series_id,
        "start": start,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": page_length,
        "offset": 0,
    }
    pages: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    total: int | None = None
    while True:
        payload = eia_get(base_url, f"{route}/data", params, settings, session)
        response = payload["response"]
        page_rows = response.get("data", [])
        total = int(response.get("total", len(page_rows)))
        pages.append(payload)
        rows.extend(page_rows)
        if not page_rows or len(rows) >= total:
            break
        params["offset"] = len(rows)
    if total is not None and len(rows) != total:
        raise DownloadError(f"EIA reported total={total} rows but {len(rows)} were received")
    if not rows:
        raise DownloadError(f"EIA returned zero rows for series {series_id}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"pages": pages}, indent=2), encoding="utf-8")
    periods = sorted(str(r.get("period")) for r in rows)
    units = sorted({str(r.get("units")) for r in rows})
    write_metadata(
        out_path,
        {
            "source": "U.S. Energy Information Administration, API v2",
            "url": f"{base_url}/{route}/data",
            "query": redact_params(params | {"offset": "paged"}),
            "series_id": series_id,
            "series_description": rows[0].get("series-description"),
            "units": units,
            "frequency": frequency,
            "downloaded_at_utc": utc_now_iso(),
            "first_period": periods[0],
            "last_period": periods[-1],
            "row_count": len(rows),
            "vintage_note": "Current-vintage values; EIA may have revised history since first publication.",
        },
    )
    log.info("Saved %d EIA rows (%s to %s) -> %s", len(rows), periods[0], periods[-1], out_path)
    return out_path


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------
def download_fred_csv(
    csv_url: str,
    series_id: str,
    out_path: Path,
    settings: HttpSettings,
    units: str,
    force: bool = False,
    session: _SessionLike | None = None,
) -> Path:
    """Download FRED's public CSV for one series and save the bytes untouched."""
    if out_path.exists() and not force:
        log.info("Using cached %s (pass --force to refresh)", out_path.name)
        return out_path
    archive_existing(out_path, out_path.parent)
    params = {"id": series_id}
    resp = http_get(csv_url, params=params, settings=settings, session=session)
    text = resp.text
    first_line = text.splitlines()[0] if text else ""
    if series_id not in first_line:
        raise APIResponseError(
            f"FRED CSV header does not mention {series_id}: {first_line[:200]!r}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(resp.content)
    lines = [ln for ln in text.splitlines()[1:] if ln.strip()]
    write_metadata(
        out_path,
        {
            "source": "Federal Reserve Bank of St. Louis, FRED (republishes EIA series RNGWHHD)",
            "url": csv_url,
            "query": params,
            "series_id": series_id,
            "units": units,
            "frequency": "daily",
            "downloaded_at_utc": utc_now_iso(),
            "first_period": lines[0].split(",")[0] if lines else None,
            "last_period": lines[-1].split(",")[0] if lines else None,
            "row_count": len(lines),
            "vintage_note": "FRED serves the latest vintage. ALFRED keeps earlier vintages.",
        },
    )
    log.info("Saved %d FRED rows -> %s", len(lines), out_path)
    return out_path


def download_fred_api(
    api_url: str,
    series_id: str,
    api_key: str,
    out_path: Path,
    settings: HttpSettings,
    units: str,
    force: bool = False,
    session: _SessionLike | None = None,
) -> Path:
    """Alternative FRED download through the keyed JSON API."""
    if out_path.exists() and not force:
        return out_path
    archive_existing(out_path, out_path.parent)
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    resp = http_get(api_url, params=params, settings=settings, session=session)
    payload = resp.json()
    if "error_message" in payload:
        raise APIResponseError(f"FRED API error: {payload['error_message']}")
    obs = payload.get("observations", [])
    if not obs:
        raise DownloadError("FRED API returned no observations")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_metadata(
        out_path,
        {
            "source": "FRED API",
            "url": api_url,
            "query": redact_params(params),
            "series_id": series_id,
            "units": units,
            "frequency": "daily",
            "downloaded_at_utc": utc_now_iso(),
            "first_period": obs[0]["date"],
            "last_period": obs[-1]["date"],
            "row_count": len(obs),
        },
    )
    return out_path
