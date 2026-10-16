from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sql_is_read_only_and_contains_required_sources() -> None:
    sql = (ROOT / "sql" / "01_extract_gate_ab_events_template.sql").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(sql.upper().split())
    assert "SET DEFAULT_TRANSACTION_READ_ONLY = ON" in normalized
    assert "CREATE TEMP" not in normalized
    assert "CREATE TABLE" not in normalized
    assert "TRUNCATE " not in normalized
    assert " DELETE " not in f" {normalized} "
    assert " UPDATE " not in f" {normalized} "
    assert "MIMICIV_DERIVED.VENTILATOR_SETTING" in normalized
    assert "MIMICIV_DERIVED.OXYGEN_DELIVERY" in normalized
    assert "MIMICIV_DERIVED.VENTILATION" in normalized
    assert "__SEED_VALUES__" in sql
