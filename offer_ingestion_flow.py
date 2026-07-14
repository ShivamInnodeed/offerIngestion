#!/usr/bin/env python3
"""LangGraph offer ingestion flow:
   fetch_data -> xml_to_json -> store_snapshot -> change_detection
   -> metadata_derivation -> normalize_data -> build_embeddings
   -> remove_stale -> index_to_es -> finalize

   Offline re-index (no XML / no LLM):
   load_updated_offers -> build_embeddings -> remove_stale -> index_to_es -> finalize
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

# Add ingestion/ to sys.path so we can import shared modules.
# In Docker the shared code lives at /app/ingestion; locally it's ../ingestion.
INGESTION_DIR = Path(os.environ.get("INGESTION_DIR", str(Path(__file__).resolve().parent.parent / "ingestion")))
if str(INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(INGESTION_DIR))

from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer

from xml_parser import (
    build_offer_markdown,
    offers_to_crawl_records,
    parse_xml_file,
    parse_xml_string,
    write_offers_json,
)
from offer_llm_enricher import (
    enrich_offers,
    load_config as load_llm_config,
)
from offer_es_mapping import ensure_offer_index
from offer_standalone_index import (
    OfferRecordNormalized,
    build_llm_lookup,
    build_records_for_pipeline,
    read_jsonl,
    to_elasticsearch_document,
    url_hash_to_offer_id_map,
    validate_and_normalize_offers,
    write_json as write_standalone_json,
    write_jsonl,
)

from es_kb.config import get_elasticsearch_client, load_settings
from es_kb.db import build_session_factory
from es_kb.audit_service import process_run_raw
from es_kb.repositories import CrawlRunRepository, SnapshotRepository, UrlRepository
from es_kb.normalize_hash import normalize_url
from build_sentence_transformer_embeddings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    iter_batches,
)
from build_embedding_payloads import load_records

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
OFFER_CONFIG_PATH = Path(os.environ.get("OFFER_CONFIG_PATH", str(_HERE / "offer_config.json")))
DEFAULT_OUTPUT_DIR = Path(os.environ.get("OFFER_OUTPUT_DIR", str(_HERE / "output")))
DEFAULT_XML_PATH = Path(os.environ.get("OFFER_XML_PATH", str(_HERE / "all_xml_data" / "all_offer_data1.xml")))
FETCH_SCRIPT_PATH = _HERE / "script.py"
LOG_DIR = Path(os.environ.get("OFFER_LOG_DIR", str(DEFAULT_OUTPUT_DIR / "logs")))

# Kept for CLI/API backward compatibility (chunking no longer used; one doc per offer).
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200

UPDATED_OFFERS_FILENAME = "updated_offers.json"
STANDALONE_OFFERS_FILENAME = "offer_standalone_normalized.json"


# ──────────────────────────────────────────────────────────────────────
# Pipeline State
# ──────────────────────────────────────────────────────────────────────

class OfferIngestionState(TypedDict, total=False):
    # API / XML input
    api_url: str | None
    xml_input_path: str | None
    xml_raw: str | None
    run_fetch_script: bool

    # Output paths
    output_dir: str
    offers_json_path: str
    crawl_records_path: str
    standalone_records_path: str
    offer_es_docs_path: str
    embeddings_output_path: str

    # Config
    config_path: str | None
    model_name: str
    batch_size: int
    chunk_size: int
    chunk_overlap: int
    recreate_index: bool
    settings_overrides: dict[str, Any]

    # LLM
    enable_llm_enrichment: bool
    max_llm_offers: int | None
    llm_output_path: str | None

    # Run tracking
    run_started_epoch_ms: int
    run_id: str | None
    is_initial_run: bool
    stage: str
    error: str | None
    timings: dict[str, float]

    # Data counts
    total_offers: int
    offers_parsed: int
    new_offers: int
    updated_offers: int
    unchanged_offers: int
    deleted_offers: int
    changed_offer_ids: list[str]
    changed_normalized_urls: list[str]
    deleted_doc_ids: list[str]
    payload_count: int
    payload_delta_count: int
    embeddings_count: int
    indexed_count: int
    deleted_count: int
    indexing_errors: int
    total_bytes: int

    # Flow control
    has_changes: bool
    stop_reason: str | None
    skip_fetch: bool
    index_from_updated_offers: bool
    updated_offers_path: str | None


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _record_error(state: OfferIngestionState, stage: str, exc: Exception) -> OfferIngestionState:
    logger.exception("Stage '%s' failed", stage)
    return {**state, "stage": stage, "error": f"{type(exc).__name__}: {exc}"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _settings_overrides_from_state(state: OfferIngestionState) -> dict[str, Any]:
    raw = state.get("settings_overrides")
    return raw if isinstance(raw, dict) else {}


def _resolve_updated_offers_path(state: OfferIngestionState) -> Path:
    explicit = state.get("updated_offers_path")
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"updated_offers file not found: {path}")
        return path
    output_dir = Path(state["output_dir"])
    for name in (UPDATED_OFFERS_FILENAME, STANDALONE_OFFERS_FILENAME):
        candidate = output_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No offers file in {output_dir}. Expected {UPDATED_OFFERS_FILENAME} "
        f"or {STANDALONE_OFFERS_FILENAME}."
    )


def _sanitize_state_for_output(state: OfferIngestionState) -> dict[str, Any]:
    out: dict[str, Any] = dict(state)
    xml_raw = out.get("xml_raw")
    if isinstance(xml_raw, str) and xml_raw:
        out["xml_raw"] = f"<omitted {len(xml_raw)} bytes>"
    return out


def _resolve_local_model_name(model_name: str) -> str:
    name = str(model_name or "").strip()
    if not name:
        return name
    local = Path("/app/models") / name
    if local.exists():
        return str(local)
    return name


# ──────────────────────────────────────────────────────────────────────
# Node 1: FETCH DATA (REST API call)
# ──────────────────────────────────────────────────────────────────────

def fetch_data_node(state: OfferIngestionState) -> OfferIngestionState:
    """Node 1: Fetch offer data.

    Three modes (checked in order):
    1. run_fetch_script — execute script.py (calls 2 internal APIs and writes XML).
    2. api_url          — direct HTTP GET (simple API).
    3. xml_input_path   — load pre-fetched XML from local file.
    4. default XML      — all_xml_data/all_offer_data1.xml (if present).

    Skipped when skip_fetch or index_from_updated_offers is set.
    """
    if state.get("error"):
        return state

    stage = "fetch_data"
    if state.get("skip_fetch") or state.get("index_from_updated_offers"):
        logger.info("Skipping XML fetch (skip_fetch / index_from_updated_offers)")
        timings = dict(state.get("timings") or {})
        timings["fetch_data_seconds"] = 0.0
        return {**state, "stage": stage, "timings": timings}

    try:
        t0 = time.perf_counter()
        output_dir = Path(state["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        run_fetch_script = bool(state.get("run_fetch_script", False))
        api_url = state.get("api_url")
        xml_input_path = state.get("xml_input_path")

        xml_raw: str | None = None

        if run_fetch_script:
            if not FETCH_SCRIPT_PATH.is_file():
                return {
                    **state,
                    "stage": stage,
                    "error": f"Fetch script not found: {FETCH_SCRIPT_PATH}",
                }

            import subprocess

            xml_out_dir = str(DEFAULT_XML_PATH.parent)
            xml_out_path = str(DEFAULT_XML_PATH)
            env = dict(os.environ)
            env["OFFER_XML_OUT_DIR"] = xml_out_dir
            env["OFFER_XML_OUT_PATH"] = xml_out_path

            logger.info("Running fetch script: %s", FETCH_SCRIPT_PATH)
            proc = subprocess.run(
                [sys.executable, str(FETCH_SCRIPT_PATH)],
                cwd=str(_HERE),
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.stdout:
                logger.info("fetch script stdout:\n%s", proc.stdout.strip())
            if proc.returncode != 0:
                logger.error("fetch script stderr:\n%s", (proc.stderr or "").strip())
                return {
                    **state,
                    "stage": stage,
                    "error": f"fetch script failed (exit={proc.returncode})",
                    "stop_reason": "fetch_script_failed",
                }

            if not DEFAULT_XML_PATH.is_file():
                return {
                    **state,
                    "stage": stage,
                    "error": f"fetch script did not produce XML at {DEFAULT_XML_PATH}",
                    "stop_reason": "fetch_script_no_output",
                }

            xml_raw = DEFAULT_XML_PATH.read_text(encoding="utf-8")
            xml_input_path = str(DEFAULT_XML_PATH)
            logger.info("Loaded fresh XML from script output: %s (%s bytes)", DEFAULT_XML_PATH, len(xml_raw))

        elif api_url:
            import requests
            logger.info("Fetching offer data from API: %s", api_url)
            resp = requests.get(api_url, timeout=120)
            resp.raise_for_status()
            xml_raw = resp.text
            if not xml_raw or not xml_raw.strip():
                return {
                    **state, "stage": stage,
                    "error": "API returned empty response",
                    "stop_reason": "api_empty",
                }
            xml_save_path = output_dir / "api_response.xml"
            xml_save_path.write_text(xml_raw, encoding="utf-8")
            logger.info("Saved API response to %s (%s bytes)", xml_save_path, len(xml_raw))

        elif xml_input_path:
            xml_file = Path(xml_input_path)
            if not xml_file.is_file():
                return {
                    **state, "stage": stage,
                    "error": f"XML input file not found: {xml_input_path}",
                }
            xml_raw = xml_file.read_text(encoding="utf-8")
            logger.info("Loaded XML from file: %s (%s bytes)", xml_input_path, len(xml_raw))

        elif DEFAULT_XML_PATH.is_file():
            xml_raw = DEFAULT_XML_PATH.read_text(encoding="utf-8")
            xml_input_path = str(DEFAULT_XML_PATH)
            logger.info(
                "No api_url or xml_input_path given — using default XML: %s (%s bytes)",
                DEFAULT_XML_PATH, len(xml_raw),
            )

        else:
            return {
                **state, "stage": stage,
                "error": "No api_url or xml_input_path provided, and default XML not found",
            }

        elapsed = time.perf_counter() - t0
        timings = dict(state.get("timings") or {})
        timings["fetch_data_seconds"] = round(elapsed, 3)
        logger.info("Fetch stage complete in %.2fs", elapsed)

        return {
            **state, "stage": stage,
            "xml_raw": xml_raw,
            "xml_input_path": xml_input_path,
            "timings": timings,
        }

    except Exception as exc:
        return _record_error(state, stage, exc)


# ──────────────────────────────────────────────────────────────────────
# Node 2: CONVERT XML TO JSON
# ──────────────────────────────────────────────────────────────────────

def xml_to_json_node(state: OfferIngestionState) -> OfferIngestionState:
    if state.get("error"):
        return state

    stage = "xml_to_json"
    try:
        t0 = time.perf_counter()
        xml_raw = state.get("xml_raw") or ""
        xml_input_path = state.get("xml_input_path")

        if xml_raw:
            offers = parse_xml_string(xml_raw)
        elif xml_input_path:
            offers = parse_xml_file(Path(xml_input_path))
        else:
            return {**state, "stage": stage, "error": "No XML data available"}

        if not offers:
            return {
                **state, "stage": stage,
                "error": "No offers parsed from XML",
                "stop_reason": "no_records",
                "total_offers": 0,
            }

        output_dir = Path(state["output_dir"])
        offers_json_path = output_dir / "offers_parsed.json"
        write_offers_json(offers, offers_json_path)

        crawl_records = offers_to_crawl_records(offers)
        crawl_json_path = output_dir / "offers_crawl_records.json"
        _write_json(crawl_json_path, {"pages": crawl_records})

        elapsed = time.perf_counter() - t0
        timings = dict(state.get("timings") or {})
        timings["xml_to_json_seconds"] = round(elapsed, 3)

        logger.info("XML→JSON: %s offers parsed in %.2fs", len(offers), elapsed)
        return {
            **state,
            "stage": stage,
            "offers_json_path": str(offers_json_path),
            "crawl_records_path": str(crawl_json_path),
            "total_offers": len(offers),
            "offers_parsed": len(offers),
            "timings": timings,
        }
    except Exception as exc:
        return _record_error(state, stage, exc)


# ──────────────────────────────────────────────────────────────────────
# Node 3: STORE RAW SNAPSHOT (SQLite: versioning, content hash, timestamp)
# ──────────────────────────────────────────────────────────────────────

def store_snapshot_node(state: OfferIngestionState) -> OfferIngestionState:
    if state.get("error"):
        return state

    stage = "store_snapshot"
    session: Any = None
    try:
        t0 = time.perf_counter()
        crawl_records_path = Path(state["crawl_records_path"])
        config_path = Path(state["config_path"]) if state.get("config_path") else OFFER_CONFIG_PATH

        crawl_records = load_records(crawl_records_path)
        settings = load_settings(config_path, overrides=_settings_overrides_from_state(state))
        session_factory = build_session_factory(settings)
        session = session_factory()

        result = process_run_raw(
            session,
            crawl_records=crawl_records,
            crawl_source_name=settings.crawl_source_name,
            root_url=settings.root_url,
            run_metadata={
                "pipeline": "offer_ingestion",
                "total_offers": state.get("total_offers", 0),
            },
        )
        session.commit()

        is_initial = (result.counters.new_urls == result.counters.total_urls and
                      result.counters.total_urls > 0 and
                      result.counters.updated_urls == 0 and
                      result.counters.unchanged_urls == 0)

        has_changes = bool(
            result.changed_normalized_urls or
            result.deleted_doc_ids or
            is_initial
        )

        elapsed = time.perf_counter() - t0
        timings = dict(state.get("timings") or {})
        timings["store_snapshot_seconds"] = round(elapsed, 3)

        logger.info(
            "Snapshot stored in %.2fs (run_id=%s new=%s updated=%s unchanged=%s deleted=%s initial=%s)",
            elapsed, result.run_id,
            result.counters.new_urls, result.counters.updated_urls,
            result.counters.unchanged_urls, result.counters.deleted_urls,
            is_initial,
        )

        return {
            **state,
            "stage": stage,
            "run_id": result.run_id,
            "is_initial_run": is_initial,
            "has_changes": has_changes,
            "changed_normalized_urls": sorted(result.changed_normalized_urls),
            "deleted_doc_ids": result.deleted_doc_ids,
            "new_offers": result.counters.new_urls,
            "updated_offers": result.counters.updated_urls,
            "unchanged_offers": result.counters.unchanged_urls,
            "deleted_offers": result.counters.deleted_urls,
            "total_bytes": result.counters.total_bytes,
            "timings": timings,
        }
    except Exception as exc:
        if session is not None:
            session.rollback()
        return _record_error(state, stage, exc)
    finally:
        if session is not None:
            session.close()


# ──────────────────────────────────────────────────────────────────────
# Node 4: CHANGE DETECTION (conditional — skip on initial run)
# ──────────────────────────────────────────────────────────────────────

def change_detection_node(state: OfferIngestionState) -> OfferIngestionState:
    if state.get("error"):
        return state

    stage = "change_detection"
    try:
        t0 = time.perf_counter()

        if state.get("is_initial_run"):
            logger.info("Initial run — all offers treated as NEW, skipping change detection")
            elapsed = time.perf_counter() - t0
            timings = dict(state.get("timings") or {})
            timings["change_detection_seconds"] = round(elapsed, 3)
            return {**state, "stage": stage, "has_changes": True, "timings": timings}

        has_changes = state.get("has_changes", False)
        changed = state.get("changed_normalized_urls") or []
        deleted = state.get("deleted_doc_ids") or []

        if not has_changes and not changed and not deleted:
            logger.info("No changes detected — pipeline will stop")
            elapsed = time.perf_counter() - t0
            timings = dict(state.get("timings") or {})
            timings["change_detection_seconds"] = round(elapsed, 3)
            return {
                **state, "stage": stage,
                "has_changes": False,
                "stop_reason": "no_changes",
                "timings": timings,
            }

        logger.info(
            "Changes detected: %s changed URLs, %s deleted doc IDs",
            len(changed), len(deleted),
        )
        elapsed = time.perf_counter() - t0
        timings = dict(state.get("timings") or {})
        timings["change_detection_seconds"] = round(elapsed, 3)
        return {**state, "stage": stage, "has_changes": True, "timings": timings}

    except Exception as exc:
        return _record_error(state, stage, exc)


# ──────────────────────────────────────────────────────────────────────
# Node 5: METADATA DERIVATION (LLM enrichment)
# ──────────────────────────────────────────────────────────────────────

def metadata_derivation_node(state: OfferIngestionState) -> OfferIngestionState:
    if state.get("error"):
        return state
    if not state.get("has_changes"):
        return state

    stage = "metadata_derivation"
    try:
        t0 = time.perf_counter()

        if not bool(state.get("enable_llm_enrichment", False)):
            logger.info("LLM enrichment disabled — skipping metadata derivation")
            return {**state, "stage": stage, "llm_output_path": None}

        offers_json_path = Path(state["offers_json_path"])
        output_dir = Path(state["output_dir"])
        llm_output_path = output_dir / "llm_enrich.json"

        all_offers = _read_json(offers_json_path)
        if not isinstance(all_offers, list):
            raise ValueError("offers_parsed.json must be a JSON array")

        changed = set(state.get("changed_normalized_urls") or [])
        if changed:
            delta_offers = [
                o for o in all_offers
                if normalize_url(str(o.get("source_url", ""))) in changed
            ]
        else:
            delta_offers = all_offers

        if not delta_offers:
            _write_json(llm_output_path, [])
            logger.info("No offers to enrich (0 delta)")
            return {**state, "stage": stage, "llm_output_path": str(llm_output_path)}

        sample_path = os.getenv("LLM_SAMPLE_PATH")
        if sample_path and Path(sample_path).exists():
            _write_json(llm_output_path, _read_json(Path(sample_path)))
            logger.info("LLM output populated from sample: %s", sample_path)
        else:
            config = load_llm_config()
            max_llm = state.get("max_llm_offers")
            results = enrich_offers(delta_offers, config=config, max_offers=max_llm)
            _write_json(llm_output_path, results)

        elapsed = time.perf_counter() - t0
        timings = dict(state.get("timings") or {})
        timings["metadata_derivation_seconds"] = round(elapsed, 3)
        logger.info("Metadata derivation complete in %.2fs", elapsed)

        return {**state, "stage": stage, "llm_output_path": str(llm_output_path), "timings": timings}

    except Exception as exc:
        return _record_error(state, stage, exc)


# ──────────────────────────────────────────────────────────────────────
# Node 6: NORMALIZE DATA (merge LLM fields back into source)
# ──────────────────────────────────────────────────────────────────────

def normalize_data_node(state: OfferIngestionState) -> OfferIngestionState:
    """Build one standalone offer record per offer (legacy offers_v1 shape)."""
    if state.get("error"):
        return state
    if not state.get("has_changes"):
        return state

    stage = "normalize_data"
    try:
        t0 = time.perf_counter()
        output_dir = Path(state["output_dir"])
        offers_json_path = Path(state["offers_json_path"])
        all_offers = _read_json(offers_json_path)
        if not isinstance(all_offers, list):
            raise ValueError("offers_parsed.json must be a JSON array")

        llm_records: list[dict[str, Any]] = []
        llm_output_path = state.get("llm_output_path")
        if llm_output_path and Path(llm_output_path).exists():
            raw_llm = _read_json(Path(llm_output_path))
            if isinstance(raw_llm, list):
                llm_records = [r for r in raw_llm if isinstance(r, dict)]

        llm_by_id, llm_by_url = build_llm_lookup(llm_records)
        changed = set(state.get("changed_normalized_urls") or [])
        filter_changed = changed if changed and not state.get("is_initial_run") else None

        raw_records = build_records_for_pipeline(
            all_offers,
            llm_by_offer_id=llm_by_id,
            llm_by_source_url=llm_by_url,
            changed_normalized_urls=filter_changed,
        )
        normalized, validation_errors = validate_and_normalize_offers(raw_records)
        if validation_errors:
            logger.warning(
                "Standalone offer validation: %s error(s), first: %s",
                len(validation_errors),
                validation_errors[0],
            )

        updated_offers_path = output_dir / "updated_offers.json"
        standalone_records_path = output_dir / "offer_standalone_normalized.json"
        normalized_payload = [o.model_dump(mode="json") for o in normalized]
        write_standalone_json(updated_offers_path, normalized_payload)
        write_standalone_json(standalone_records_path, normalized_payload)

        elapsed = time.perf_counter() - t0
        timings = dict(state.get("timings") or {})
        timings["normalize_data_seconds"] = round(elapsed, 3)

        logger.info(
            "Normalize data complete in %.2fs (offers=%s, validation_errors=%s)",
            elapsed,
            len(normalized),
            len(validation_errors),
        )
        return {
            **state,
            "stage": stage,
            "standalone_records_path": str(standalone_records_path),
            "payload_count": len(normalized),
            "payload_delta_count": len(normalized),
            "timings": timings,
        }

    except Exception as exc:
        return _record_error(state, stage, exc)


# ──────────────────────────────────────────────────────────────────────
# Offline path: LOAD UPDATED OFFERS (skip XML / LLM / change detection)
# ──────────────────────────────────────────────────────────────────────

def load_updated_offers_node(state: OfferIngestionState) -> OfferIngestionState:
    """Load pre-built updated_offers.json from output_dir and prepare for embed + index."""
    if state.get("error"):
        return state

    stage = "load_updated_offers"
    try:
        t0 = time.perf_counter()
        records_path = _resolve_updated_offers_path(state)
        raw = _read_json(records_path)
        if not isinstance(raw, list) or not raw:
            return {
                **state,
                "stage": stage,
                "error": f"No offers in {records_path}",
                "stop_reason": "no_offers_in_updated_json",
            }

        offer_count = len([row for row in raw if isinstance(row, dict)])
        if offer_count == 0:
            return {
                **state,
                "stage": stage,
                "error": f"No valid offer objects in {records_path}",
                "stop_reason": "no_offers_in_updated_json",
            }

        elapsed = time.perf_counter() - t0
        timings = dict(state.get("timings") or {})
        timings["load_updated_offers_seconds"] = round(elapsed, 3)

        logger.info(
            "Offline index mode: loaded %s offer(s) from %s (skip fetch/XML/LLM/change detection)",
            offer_count,
            records_path,
        )

        return {
            **state,
            "stage": stage,
            "standalone_records_path": str(records_path),
            "updated_offers_path": str(records_path),
            "has_changes": True,
            "is_initial_run": True,
            "enable_llm_enrichment": False,
            "total_offers": offer_count,
            "offers_parsed": offer_count,
            "payload_count": offer_count,
            "payload_delta_count": offer_count,
            "timings": timings,
            "stop_reason": None,
        }
    except Exception as exc:
        return _record_error(state, stage, exc)


# ──────────────────────────────────────────────────────────────────────
# Node 7: BUILD EMBEDDINGS (embed offer summary using embedding model)
# ──────────────────────────────────────────────────────────────────────

def build_embeddings_node(state: OfferIngestionState) -> OfferIngestionState:
    """One embedding per offer on searchable_text (legacy standalone_ingest)."""
    if state.get("error"):
        return state
    if not state.get("has_changes"):
        return state

    stage = "build_embeddings"
    try:
        t0 = time.perf_counter()
        output_dir = Path(state["output_dir"])
        records_path = Path(state["standalone_records_path"])
        offer_es_docs_path = output_dir / "offer_es_docs.jsonl"
        embeddings_output_path = Path(state["embeddings_output_path"])
        batch_size = int(state.get("batch_size") or DEFAULT_BATCH_SIZE)

        raw = _read_json(records_path)
        if not isinstance(raw, list) or not raw:
            write_jsonl(offer_es_docs_path, [])
            write_jsonl(embeddings_output_path, [])
            elapsed = time.perf_counter() - t0
            timings = dict(state.get("timings") or {})
            timings["embedding_seconds"] = round(elapsed, 3)
            logger.info("Embedding stage skipped (0 offers)")
            return {
                **state,
                "stage": stage,
                "offer_es_docs_path": str(offer_es_docs_path),
                "embeddings_count": 0,
                "timings": timings,
            }

        normalized: list[OfferRecordNormalized] = [
            OfferRecordNormalized.model_validate(item) for item in raw if isinstance(item, dict)
        ]
        texts = [o.searchable_text for o in normalized if o.searchable_text.strip()]
        if len(texts) != len(normalized):
            logger.warning("Some offers have empty searchable_text and will be skipped")

        model = SentenceTransformer(_resolve_local_model_name(state["model_name"]))
        es_docs: list[dict[str, Any]] = []
        embedding_debug: list[dict[str, Any]] = []
        valid_offers = [o for o in normalized if o.searchable_text.strip()]

        for batch_num, batch in enumerate(iter_batches(valid_offers, batch_size), start=1):
            batch_texts = [o.searchable_text for o in batch]
            logger.info("Encoding offer batch %s (%s records)", batch_num, len(batch))
            vectors = model.encode(
                batch_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for offer, emb in zip(batch, vectors):
                vec = emb.tolist()
                doc = to_elasticsearch_document(offer, vec)
                es_docs.append(doc)
                embedding_debug.append(
                    {
                        "embedding_text": offer.searchable_text,
                        "embedding": vec,
                        "metadata": {"offer_id": offer.offer_id, "ast_path": offer.ast_path},
                    }
                )

        write_jsonl(offer_es_docs_path, es_docs)
        write_jsonl(embeddings_output_path, embedding_debug)

        elapsed = time.perf_counter() - t0
        timings = dict(state.get("timings") or {})
        timings["embedding_seconds"] = round(elapsed, 3)
        logger.info(
            "Embedding stage complete in %.2fs (%s offers → %s ES docs)",
            elapsed,
            len(valid_offers),
            len(es_docs),
        )

        return {
            **state,
            "stage": stage,
            "offer_es_docs_path": str(offer_es_docs_path),
            "embeddings_count": len(es_docs),
            "timings": timings,
        }
    except Exception as exc:
        return _record_error(state, stage, exc)


# ──────────────────────────────────────────────────────────────────────
# Node 8: REMOVE STALE (remove ES index of deleted/updated offers)
# ──────────────────────────────────────────────────────────────────────

def remove_stale_node(state: OfferIngestionState) -> OfferIngestionState:
    if state.get("error"):
        return state
    if not state.get("has_changes"):
        return state

    stage = "remove_stale"
    try:
        t0 = time.perf_counter()

        if state.get("is_initial_run"):
            logger.info("Initial run — no stale docs to remove")
            elapsed = time.perf_counter() - t0
            timings = dict(state.get("timings") or {})
            timings["remove_stale_seconds"] = round(elapsed, 3)
            return {**state, "stage": stage, "deleted_count": 0, "timings": timings}

        deleted_doc_ids = list(state.get("deleted_doc_ids") or [])
        if not deleted_doc_ids:
            logger.info("No stale documents to remove")
            elapsed = time.perf_counter() - t0
            timings = dict(state.get("timings") or {})
            timings["remove_stale_seconds"] = round(elapsed, 3)
            return {**state, "stage": stage, "deleted_count": 0, "timings": timings}

        config_path = Path(state["config_path"]) if state.get("config_path") else OFFER_CONFIG_PATH
        settings = load_settings(config_path, overrides=_settings_overrides_from_state(state))
        es = get_elasticsearch_client(settings)

        # Audit stores URL-hash doc ids; legacy offers_v1 index uses offer_id as _id.
        offers_json_path = state.get("offers_json_path")
        url_to_offer: dict[str, str] = {}
        if offers_json_path and Path(offers_json_path).exists():
            all_offers = _read_json(Path(offers_json_path))
            if isinstance(all_offers, list):
                url_to_offer = url_hash_to_offer_id_map(all_offers)

        es_doc_ids: list[str] = []
        for doc_id in deleted_doc_ids:
            es_doc_ids.append(url_to_offer.get(doc_id, doc_id))

        deleted = 0
        from elasticsearch.helpers import bulk as es_bulk
        from datetime import datetime, timezone

        if settings.es_delete_mode == "hard_delete":
            actions = [
                {"_op_type": "delete", "_index": settings.index_name, "_id": doc_id}
                for doc_id in es_doc_ids
            ]
        else:
            deleted_at = datetime.now(timezone.utc).isoformat()
            actions = [
                {
                    "_op_type": "update", "_index": settings.index_name, "_id": doc_id,
                    "doc": {"is_active": False, "deleted_at": deleted_at, "change_status": "DELETED"},
                    "doc_as_upsert": False,
                }
                for doc_id in es_doc_ids
            ]

        ok, errors = es_bulk(es, actions, chunk_size=settings.bulk_chunk_size,
                             raise_on_error=False, refresh=settings.bulk_refresh)
        deleted = int(ok or 0)
        error_count = len(errors) if errors else 0
        if error_count:
            logger.warning("Stale removal had %s errors", error_count)

        elapsed = time.perf_counter() - t0
        timings = dict(state.get("timings") or {})
        timings["remove_stale_seconds"] = round(elapsed, 3)
        logger.info("Remove stale complete in %.2fs (deleted=%s)", elapsed, deleted)

        return {**state, "stage": stage, "deleted_count": deleted, "timings": timings}

    except Exception as exc:
        return _record_error(state, stage, exc)


# ──────────────────────────────────────────────────────────────────────
# Node 9: INDEX TO ES (legacy offers_v1 document shape)
# ──────────────────────────────────────────────────────────────────────

def index_to_es_node(state: OfferIngestionState) -> OfferIngestionState:
    if state.get("error"):
        return state
    if not state.get("has_changes"):
        return state

    stage = "index_to_es"
    try:
        t0 = time.perf_counter()
        config_path = Path(state["config_path"]) if state.get("config_path") else OFFER_CONFIG_PATH
        offer_es_docs_path = Path(state["offer_es_docs_path"])

        settings = load_settings(config_path, overrides=_settings_overrides_from_state(state))
        es = get_elasticsearch_client(settings)

        ensure_offer_index(es, settings, recreate=bool(state.get("recreate_index", False)))

        es_docs = read_jsonl(offer_es_docs_path)
        from elasticsearch.helpers import bulk as es_bulk

        def _build_actions():
            for doc in es_docs:
                offer_id = str(doc.get("offer_id") or "").strip()
                if not offer_id:
                    logger.warning("Skipping ES doc without offer_id")
                    continue
                yield {
                    "_op_type": "index",
                    "_index": settings.index_name,
                    "_id": offer_id,
                    "_source": doc,
                }

        ok, errors = es_bulk(
            es,
            _build_actions(),
            chunk_size=settings.bulk_chunk_size,
            raise_on_error=False,
            refresh=settings.bulk_refresh,
        )
        error_count = len(errors) if errors else 0
        if error_count:
            for idx, err in enumerate((errors or [])[:3], start=1):
                logger.error("ES bulk error #%s: %s", idx, err)

        if settings.bulk_refresh is False:
            try:
                es.indices.refresh(index=settings.index_name)
            except Exception:
                logger.debug("Index refresh skipped", exc_info=True)

        run_id = state.get("run_id")
        if run_id:
            try:
                db_session = build_session_factory(settings)()
                indexed_urls = []
                for doc in es_docs:
                    ast = str(doc.get("ast_path") or "").strip()
                    norm = normalize_url(ast)
                    if norm:
                        indexed_urls.append(norm)
                if indexed_urls:
                    SnapshotRepository(db_session).mark_indexed_for_run(run_id, indexed_urls)
                    UrlRepository(db_session).mark_indexed_for_normalized_urls(indexed_urls)
                    db_session.commit()
                    logger.info("Marked %s URLs as INDEXED in SQLite", len(indexed_urls))
                db_session.close()
            except Exception:
                logger.exception("Failed to mark INDEXED in SQLite")

        elapsed = time.perf_counter() - t0
        timings = dict(state.get("timings") or {})
        timings["index_seconds"] = round(elapsed, 3)

        logger.info(
            "ES index complete in %.2fs (indexed=%s errors=%s index=%s)",
            elapsed,
            ok,
            error_count,
            settings.index_name,
        )
        return {
            **state,
            "stage": stage,
            "indexed_count": int(ok or 0),
            "indexing_errors": error_count,
            "timings": timings,
        }
    except Exception as exc:
        return _record_error(state, stage, exc)


# ──────────────────────────────────────────────────────────────────────
# Node 10: FINALIZE RUN
# ──────────────────────────────────────────────────────────────────────

def finalize_node(state: OfferIngestionState) -> OfferIngestionState:
    run_id = state.get("run_id")
    if not run_id:
        logger.warning("finalize skipped: no run_id")
        return {**state, "stage": "finalize"}

    stage = "finalize"
    session: Any = None
    try:
        config_path = Path(state["config_path"]) if state.get("config_path") else OFFER_CONFIG_PATH
        settings = load_settings(config_path, overrides=_settings_overrides_from_state(state))
        session_factory = build_session_factory(settings)
        session = session_factory()

        elapsed_ms = int(time.time() * 1000) - int(state.get("run_started_epoch_ms") or int(time.time() * 1000))
        pipeline_error = state.get("error")
        idx_err = int(state.get("indexing_errors") or 0)

        if pipeline_error:
            run_status = "FAILED"
        elif idx_err > 0:
            run_status = "PARTIAL"
        else:
            run_status = "COMPLETED"

        stop_reason = state.get("stop_reason")
        if stop_reason and not pipeline_error:
            run_status = "COMPLETED"

        failed_reason = None
        if run_status == "FAILED" and pipeline_error:
            failed_reason = str(pipeline_error)
        elif run_status == "PARTIAL":
            failed_reason = f"PARTIAL: es_errors={idx_err}"

        run_repo = CrawlRunRepository(session)
        run_repo.finalize_run(
            run_id=run_id,
            run_status=run_status,
            total_urls=int(state.get("total_offers") or 0),
            success_urls=int(state.get("total_offers") or 0),
            failed_urls=0,
            new_urls=int(state.get("new_offers") or 0),
            updated_urls=int(state.get("updated_offers") or 0),
            unchanged_urls=int(state.get("unchanged_offers") or 0),
            deleted_urls=int(state.get("deleted_offers") or 0),
            total_bytes=int(state.get("total_bytes") or 0),
            duration_ms=elapsed_ms,
            failed_reason=failed_reason,
            outcome_metadata={
                "pipeline": "offer_ingestion",
                "indexed_count": int(state.get("indexed_count") or 0),
                "indexing_errors": idx_err,
                "deleted_count": int(state.get("deleted_count") or 0),
                "embeddings_count": int(state.get("embeddings_count") or 0),
                "stop_reason": stop_reason,
                "is_initial_run": bool(state.get("is_initial_run")),
            },
        )
        session.commit()
        logger.info("Run finalized: status=%s run_id=%s", run_status, run_id)
        return {**state, "stage": stage}

    except Exception as exc:
        if session is not None:
            session.rollback()
        return _record_error(state, stage, exc)
    finally:
        if session is not None:
            session.close()


# ──────────────────────────────────────────────────────────────────────
# Routing functions for conditional edges
# ──────────────────────────────────────────────────────────────────────

def _route_from_start(state: OfferIngestionState) -> str:
    if state.get("error"):
        return "finalize"
    if state.get("index_from_updated_offers"):
        return "load_updated_offers"
    return "fetch_data"


def _should_continue_after_change_detection(state: OfferIngestionState) -> str:
    """After change detection: proceed or stop."""
    if state.get("error"):
        return "finalize"
    if not state.get("has_changes"):
        return "finalize"
    return "metadata_derivation"


def _should_continue_after_fetch(state: OfferIngestionState) -> str:
    """After fetch: stop on error or continue."""
    if state.get("error"):
        return "finalize"
    return "xml_to_json"


def _should_continue_after_xml(state: OfferIngestionState) -> str:
    if state.get("error"):
        return "finalize"
    return "store_snapshot"


# ──────────────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(OfferIngestionState)

    graph.add_node("fetch_data", fetch_data_node)
    graph.add_node("load_updated_offers", load_updated_offers_node)
    graph.add_node("xml_to_json", xml_to_json_node)
    graph.add_node("store_snapshot", store_snapshot_node)
    graph.add_node("change_detection", change_detection_node)
    graph.add_node("metadata_derivation", metadata_derivation_node)
    graph.add_node("normalize_data", normalize_data_node)
    graph.add_node("build_embeddings", build_embeddings_node)
    graph.add_node("remove_stale", remove_stale_node)
    graph.add_node("index_to_es", index_to_es_node)
    graph.add_node("finalize", finalize_node)

    graph.add_conditional_edges(START, _route_from_start)
    graph.add_conditional_edges("fetch_data", _should_continue_after_fetch)
    graph.add_conditional_edges("xml_to_json", _should_continue_after_xml)
    graph.add_edge("store_snapshot", "change_detection")
    graph.add_conditional_edges("change_detection", _should_continue_after_change_detection)
    graph.add_edge("load_updated_offers", "build_embeddings")
    graph.add_edge("metadata_derivation", "normalize_data")
    graph.add_edge("normalize_data", "build_embeddings")
    graph.add_edge("build_embeddings", "remove_stale")
    graph.add_edge("remove_stale", "index_to_es")
    graph.add_edge("index_to_es", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def run_offer_ingestion(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_path: Path | None = None,
    api_url: str | None = None,
    xml_input_path: str | None = None,
    run_fetch_script: bool = False,
    skip_fetch: bool = False,
    index_from_updated_offers: bool = False,
    updated_offers_path: str | None = None,
    index_name: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    recreate_index: bool = False,
    enable_llm_enrichment: bool = False,
    max_llm_offers: int | None = None,
    es_url: str | None = None,
    es_username: str | None = None,
    es_password: str | None = None,
    db_url: str | None = None,
) -> OfferIngestionState:
    app = build_graph()

    settings_overrides: dict[str, Any] = {
        "elasticsearch_url": es_url or os.getenv("DEFAULT_ES_URL") or os.getenv("ES_ELASTICSEARCH_URL"),
        "elasticsearch_username": es_username,
        "elasticsearch_password": es_password,
        "db_url": db_url,
        "index_name": index_name,
    }

    if index_from_updated_offers:
        skip_fetch = True
        enable_llm_enrichment = False
        run_fetch_script = False

    initial: OfferIngestionState = {
        "api_url": api_url,
        "xml_input_path": xml_input_path,
        "run_fetch_script": bool(run_fetch_script),
        "skip_fetch": bool(skip_fetch),
        "index_from_updated_offers": bool(index_from_updated_offers),
        "updated_offers_path": updated_offers_path,
        "output_dir": str(output_dir),
        "config_path": str(config_path) if config_path else str(OFFER_CONFIG_PATH),
        "model_name": model_name,
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "recreate_index": recreate_index,
        "enable_llm_enrichment": enable_llm_enrichment,
        "max_llm_offers": max_llm_offers,
        "settings_overrides": settings_overrides,
        "embeddings_output_path": str(output_dir / "embedding_vectors.jsonl"),
        "standalone_records_path": str(output_dir / "offer_standalone_normalized.json"),
        "offer_es_docs_path": str(output_dir / "offer_es_docs.jsonl"),
        "run_started_epoch_ms": int(time.time() * 1000),
        "timings": {},
        "error": None,
        "stage": "init",
        "deleted_doc_ids": [],
        "changed_normalized_urls": [],
        "has_changes": False,
        "stop_reason": None,
    }
    return app.invoke(initial)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SBI Card Offer Ingestion Pipeline (LangGraph)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=None, help="Path to offer config JSON.")
    parser.add_argument("--api-url", default=None, help="REST API URL to fetch offer XML.")
    parser.add_argument("--xml-input", default=None, help="Path to local XML file (alternative to API).")
    parser.add_argument("--run-fetch-script", action="store_true", help="Run script.py to fetch fresh XML before parsing.")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip XML fetch (use with --index-from-updated-offers for offline re-index).",
    )
    parser.add_argument(
        "--index-from-updated-offers",
        action="store_true",
        help="Index from output/updated_offers.json (skip fetch, XML, LLM, change detection).",
    )
    parser.add_argument(
        "--updated-offers-file",
        default=None,
        help="Explicit path to updated_offers.json (default: output-dir/updated_offers.json).",
    )
    parser.add_argument(
        "--index-name",
        default=None,
        help="Elasticsearch index name override (e.g. kb_documents_offers).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--recreate-index", action="store_true")
    parser.add_argument("--enable-llm-enrichment", action="store_true")
    parser.add_argument("--max-llm-offers", type=int, default=None)
    parser.add_argument(
        "--es-url",
        default=os.getenv("DEFAULT_ES_URL") or os.getenv("ES_ELASTICSEARCH_URL"),
        help="Override ES URL.",
    )
    parser.add_argument("--es-username", default=None)
    parser.add_argument("--es-password", default=None)
    parser.add_argument("--db-url", default=None, help="Override SQLAlchemy DB URL.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output state JSON.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _setup_logging(level: int) -> None:
    """Configure root logger to write to both stderr and a rotating log file."""
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime as _dt
    log_file = LOG_DIR / f"offer_ingestion_{_dt.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)
    logger.info("Logging to %s", log_file)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    _setup_logging(getattr(logging, args.log_level))

    try:
        final_state = run_offer_ingestion(
            output_dir=args.output_dir,
            config_path=args.config,
            api_url=args.api_url,
            xml_input_path=args.xml_input,
            run_fetch_script=bool(args.run_fetch_script),
            skip_fetch=bool(args.skip_fetch),
            index_from_updated_offers=bool(args.index_from_updated_offers),
            updated_offers_path=args.updated_offers_file,
            index_name=args.index_name,
            model_name=args.model_name,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            recreate_index=args.recreate_index,
            enable_llm_enrichment=args.enable_llm_enrichment,
            max_llm_offers=args.max_llm_offers,
            es_url=args.es_url,
            es_username=args.es_username,
            es_password=args.es_password,
            db_url=args.db_url,
        )
    except Exception:
        logger.exception("Offer ingestion flow failed")
        return 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(final_state, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote flow state to %s", args.output)

    print(json.dumps(_sanitize_state_for_output(final_state), ensure_ascii=False, indent=2))

    if final_state.get("error"):
        return 2
    if int(final_state.get("indexing_errors") or 0) > 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
