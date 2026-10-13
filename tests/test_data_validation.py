"""Parsing and validation tests using small SYNTHETIC files written to a temp folder."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from natgas_quant import data_download as DD
from natgas_quant import data_validation as DV


def _write(text: str, name: str = "f.csv") -> Path:
    d = Path(tempfile.mkdtemp())
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# FRED CSV
# ---------------------------------------------------------------------------
def test_parse_fred_new_header_and_empty_missing():
    p = _write("observation_date,DHHNGSP\n2024-01-02,2.50\n2024-01-03,\n2024-01-04,2.61\n")
    df = DV.parse_fred_csv(p)
    assert df["date_raw"].tolist()[0] == "2024-01-02"
    assert math.isnan(df["price"].iloc[1])
    assert df["price"].iloc[2] == 2.61


def test_parse_fred_old_header_and_dot_missing():
    df = DV.parse_fred_csv(_write("DATE,DHHNGSP\n2001-01-02,9.98\n2001-01-03,.\n"))
    assert math.isnan(df["price"].iloc[1])
    assert df["value_raw"].iloc[1] == "."


def test_unknown_symbol_raises_instead_of_becoming_nan():
    with pytest.raises(DV.DataValidationError):
        DV.parse_fred_csv(_write("DATE,DHHNGSP\n2001-01-02,abc\n"))


def _prices(dates, values):
    return pd.DataFrame({"date": pd.to_datetime(dates), "price": values,
                         "date_raw": dates, "value_raw": [str(v) for v in values]})


def test_duplicate_dates_rejected():
    with pytest.raises(DV.DataValidationError):
        DV.validate_prices(_prices(["2024-01-02", "2024-01-02"], [2.0, 2.1]))


def test_nonpositive_prices_rejected():
    with pytest.raises(DV.DataValidationError):
        DV.validate_prices(_prices(["2024-01-02", "2024-01-03"], [2.0, 0.0]))


def test_validate_prices_sorts_and_reports():
    df, rep = DV.validate_prices(_prices(["2024-01-03", "2024-01-02", "2024-01-05"], [2.1, 2.0, np.nan]))
    assert df["date"].is_monotonic_increasing
    assert rep.facts["rows_with_missing_price"] == 1
    assert rep.facts["business_days_absent_from_file"] == 1   # 2024-01-04
    assert rep.facts["input_was_sorted"] is False
    assert len(DV.missing_price_table(df)) == 1


def test_empty_price_table_rejected():
    with pytest.raises(DV.DataValidationError):
        DV.validate_prices(_prices([], []))


# ---------------------------------------------------------------------------
# EIA storage
# ---------------------------------------------------------------------------
def _eia_file(rows):
    payload = {"pages": [{"response": {"total": len(rows), "data": rows}}]}
    return _write(json.dumps(payload), "eia.json")


def _row(period, value, units="BCF", series="NW2_EPG0_SWO_R48_BCF"):
    return {"period": period, "value": value, "units": units, "series": series}


def test_storage_parse_and_validate():
    p = _eia_file([_row("2024-01-12", "3336"), _row("2024-01-05", "3490"), _row("2024-01-19", "3010")])
    raw = DV.parse_eia_storage_json(p)
    df, rep = DV.validate_storage(raw, "NW2_EPG0_SWO_R48_BCF")
    assert df["week_ending_date"].is_monotonic_increasing
    assert df["storage_level_bcf"].tolist() == [3490.0, 3336.0, 3010.0]
    assert rep.facts["non_friday_week_endings"] == 0
    assert "period_raw" in df.columns


def test_storage_wrong_units_rejected():
    raw = DV.parse_eia_storage_json(_eia_file([_row("2024-01-05", "100", units="MMCF")]))
    with pytest.raises(DV.DataValidationError):
        DV.validate_storage(raw, "NW2_EPG0_SWO_R48_BCF")


def test_storage_wrong_series_rejected():
    raw = DV.parse_eia_storage_json(_eia_file([_row("2024-01-05", "100", series="OTHER")]))
    with pytest.raises(DV.DataValidationError):
        DV.validate_storage(raw, "NW2_EPG0_SWO_R48_BCF")


def test_storage_irregular_spacing_warned():
    raw = DV.parse_eia_storage_json(_eia_file([_row("2024-01-05", "1"), _row("2024-01-19", "2")]))
    _, rep = DV.validate_storage(raw, "NW2_EPG0_SWO_R48_BCF")
    assert rep.facts["irregular_spacing_count"] == 1


def test_compare_vintages_finds_revision():
    old = pd.DataFrame({"d": [1, 2, 3], "v": [10.0, 20.0, 30.0]})
    new = pd.DataFrame({"d": [1, 2, 3], "v": [10.0, 21.0, 30.0]})
    rev = DV.compare_vintages(old, new, "d", "v")
    assert rev["d"].tolist() == [2] and rev["revision"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Download helpers (no network: fake session)
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or json.dumps(payload or {})
        self.content = self.text.encode()

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        return self.responses.pop(0)


def test_http_retries_transient_errors_then_succeeds():
    s = _Session([_Resp(503), _Resp(200, {"ok": 1})])
    r = DD.http_get("https://x", session=s, settings=DD.HttpSettings(max_retries=2, backoff_seconds=0),
                    sleep=lambda _: None)
    assert r.status_code == 200 and s.calls == 2


def test_http_does_not_retry_client_errors():
    s = _Session([_Resp(400, text="bad request")])
    with pytest.raises(DD.APIResponseError):
        DD.http_get("https://x", session=s, settings=DD.HttpSettings(max_retries=3, backoff_seconds=0),
                    sleep=lambda _: None)
    assert s.calls == 1


def test_eia_error_payload_raises():
    with pytest.raises(DD.APIResponseError):
        DD.check_eia_payload({"error": "Invalid frequency", "code": 400})


def test_api_key_is_redacted():
    out = DD.redact_params({"api_key": "SECRET", "frequency": "weekly"})
    assert out["api_key"] != "SECRET" and out["frequency"] == "weekly"


def test_find_facet_value_raises_when_absent():
    payload = {"response": {"facets": [{"id": "A"}, {"id": "B"}]}}
    assert DD.find_facet_value(payload, "B")["id"] == "B"
    with pytest.raises(DD.DownloadError):
        DD.find_facet_value(payload, "C")
