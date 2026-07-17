#!/usr/bin/env python3
"""Bootstrap the empty Oracle metadata schema for offer ingestion.

Usage:
  ES_DB_URL='oracle+oracledb://GENIE_APP:user%23123ab@host:1875/?service_name=GENIEUAT_N' \\
    python bootstrap_oracle_schema.py

Or:
  python bootstrap_oracle_schema.py --db-url 'oracle+oracledb://...'
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

INGESTION_DIR = Path(
    os.environ.get("INGESTION_DIR", str(Path(__file__).resolve().parent.parent / "ingestion"))
)
if str(INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(INGESTION_DIR))

from es_kb.config import load_settings
from es_kb.db import (
    create_engine_from_settings,
    init_db,
    redact_db_url,
    verify_oracle_connection,
)
from es_kb.db_models import Base

logger = logging.getLogger(__name__)

DEFAULT_EXPECTED_USER = "GENIE_APP"
OFFER_CONFIG_PATH = Path(os.environ.get("OFFER_CONFIG_PATH", str(Path(__file__).resolve().parent / "offer_config.json")))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Create empty Oracle audit tables for offer ingestion.")
    parser.add_argument("--config", type=Path, default=OFFER_CONFIG_PATH, help="Path to offer_config.json")
    parser.add_argument("--db-url", default=None, help="Override SQLAlchemy DB URL (prefer ES_DB_URL env).")
    parser.add_argument(
        "--expected-user",
        default=DEFAULT_EXPECTED_USER,
        help=f"Assert connected Oracle schema user (default: {DEFAULT_EXPECTED_USER}).",
    )
    parser.add_argument(
        "--skip-user-check",
        action="store_true",
        help="Do not assert the connected Oracle user.",
    )
    args = parser.parse_args()

    overrides = {}
    if args.db_url:
        overrides["db_url"] = args.db_url
    settings = load_settings(args.config, overrides=overrides or None)
    if not (settings.db_url or "").strip():
        logger.error("No db_url configured. Set ES_DB_URL or pass --db-url.")
        return 1

    logger.info("Connecting with %s", redact_db_url(settings.db_url))
    engine = create_engine_from_settings(settings)

    if settings.db_url.strip().startswith("oracle"):
        expected = None if args.skip_user_check else args.expected_user
        current_user = verify_oracle_connection(engine, expected_user=expected)
        logger.info("Connected as Oracle user %s", current_user)
    else:
        logger.warning("db_url is not Oracle; creating schema for %s", redact_db_url(settings.db_url))

    init_db(engine)
    table_names = sorted(Base.metadata.tables.keys())
    logger.info("Schema ready. Tables: %s", ", ".join(table_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
