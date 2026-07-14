"""Elasticsearch index mapping for SBI Card offers (legacy offers_v1 / standalone_ingest)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch

INGESTION_DIR = Path(__file__).resolve().parent.parent / "ingestion"
if str(INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(INGESTION_DIR))

from es_kb.config import AppSettings

from offer_standalone_index import build_standalone_index_body


def build_offer_index_body(settings: AppSettings) -> dict[str, Any]:
    return build_standalone_index_body(settings)


def ensure_offer_index(es: Elasticsearch, settings: AppSettings, recreate: bool = False) -> None:
    name = settings.index_name
    if recreate and es.indices.exists(index=name):
        es.indices.delete(index=name)
    if es.indices.exists(index=name):
        return
    body = build_offer_index_body(settings)
    es.indices.create(index=name, settings=body["settings"], mappings=body["mappings"])
