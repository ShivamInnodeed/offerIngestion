from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

ROOT = Path(__file__).resolve().parents[1]
INGESTION = Path(os.environ.get("INGESTION_DIR", str(ROOT.parent / "ingestion")))
if str(INGESTION) not in sys.path:
    sys.path.insert(0, str(INGESTION))

from es_kb.config import AppSettings, load_settings
from es_kb.db import (
    build_oracle_db_url,
    build_session_factory,
    create_engine_from_settings,
    init_db,
    redact_db_url,
)
from es_kb.db_models import Base, CrawlRun, UrlMaster
from es_kb.oracle_types import format_oracle_timestamp_tz, parse_oracle_timestamp_tz
from es_kb.repositories import CrawlRunRepository, UrlRepository


def test_redact_db_url_strips_password() -> None:
    url = "oracle+oracledb://GENIE_APP:user%23123ab@host.example:1875/?service_name=GENIEUAT_N"
    redacted = redact_db_url(url)
    assert redacted == "oracle+oracledb://host.example:1875/?service_name=GENIEUAT_N"


def test_build_oracle_db_url_encodes_hash_password() -> None:
    url = build_oracle_db_url(
        user="GENIE_APP",
        password="user#123ab",
        host="sbigenieuat-scan.sbic.sbicard.com",
        port=1875,
        service_name="GENIEUAT_N",
    )
    assert url.startswith("oracle+oracledb://")
    assert "user%23123ab" in url
    assert "service_name=GENIEUAT_N" in url


def test_load_settings_prefers_es_db_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "offer_config.json"
    cfg.write_text('{"db_url": "", "index_name": "kb_documents_offers"}', encoding="utf-8")
    monkeypatch.setenv(
        "ES_DB_URL",
        "oracle+oracledb://GENIE_APP:user%23123ab@host:1875/?service_name=GENIEUAT_N",
    )
    settings = load_settings(cfg)
    assert settings.db_url.startswith("oracle+oracledb://")
    assert settings.index_name == "kb_documents_offers"


def test_timestamp_round_trip_text() -> None:
    dt = datetime(2026, 6, 16, 11, 39, 21, 343593, tzinfo=timezone.utc)
    text = format_oracle_timestamp_tz(dt)
    assert text == "16-JUN-2026 11:39:21.343593 AM +00:00"
    parsed = parse_oracle_timestamp_tz(text)
    assert parsed == dt


def test_sqlite_session_factory_creates_schema_by_default(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    settings = AppSettings(db_url=f"sqlite:///{db_path}")
    factory = build_session_factory(settings)
    engine = factory.kw["bind"]
    assert engine.dialect.name == "sqlite"
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='website_crawl_run'"
        ).fetchall()
    assert rows

    session = factory()
    try:
        run = CrawlRunRepository(session).create_run(
            crawl_source_name="sbicard_offers",
            root_url="https://www.sbicard.com",
        )
        session.commit()
        assert run.run_id
        loaded = session.get(CrawlRun, run.run_id)
        assert loaded is not None
        assert loaded.crawl_source_name == "sbicard_offers"
    finally:
        session.close()


def test_sqlite_json_and_boolean_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "json.db"
    settings = AppSettings(db_url=f"sqlite:///{db_path}")
    engine = create_engine_from_settings(settings)
    init_db(engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    session: Session = SessionLocal()
    try:
        repo = CrawlRunRepository(session)
        run = repo.create_run(
            crawl_source_name="sbicard_offers",
            root_url="https://www.sbicard.com",
            run_metadata={"pipeline": "offer_ingestion", "total_offers": 2},
        )
        url_repo = UrlRepository(session)
        master = url_repo.upsert_url(
            normalized_url="https://www.sbicard.com/offers/test",
            source_url="https://www.sbicard.com/offers/test",
            domain_name="www.sbicard.com",
            canonical_url=None,
            now=datetime.now(timezone.utc),
        )
        session.commit()

        loaded_run = session.execute(select(CrawlRun).where(CrawlRun.run_id == run.run_id)).scalar_one()
        assert loaded_run.run_metadata_json == {"pipeline": "offer_ingestion", "total_offers": 2}

        loaded_url = session.execute(
            select(UrlMaster).where(UrlMaster.normalized_url == master.normalized_url)
        ).scalar_one()
        assert loaded_url.is_active is True
    finally:
        session.close()


def test_oracle_timestamp_type_compiles() -> None:
    from sqlalchemy.dialects import oracle

    dialect = oracle.dialect()
    col = CrawlRun.__table__.c.started_at
    compiled = col.type.compile(dialect=dialect)
    assert "TIMESTAMP" in str(compiled).upper()
