"""Standalone offer indexing helpers (aligned with standalone_ingest.py).

- One Elasticsearch document per offer (no markdown chunking).
- Embedding on searchable_text with L2-normalized vectors (cosine).
- ES mapping and document shape match legacy offers_v1 index.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field, field_validator

_INGESTION_DIR = Path(__file__).resolve().parent.parent / "ingestion"
if str(_INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(_INGESTION_DIR))

from es_kb.config import AppSettings
from es_kb.normalize_hash import build_doc_id, normalize_url

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSIONS = 384

# ---------------------------------------------------------------------------
# Elasticsearch index (standalone_ingest.py)
# ---------------------------------------------------------------------------
INDEX_SETTINGS = {
    "index": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "5s",
    },
    "analysis": {
        "analyzer": {
            "offer_text": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "asciifolding"],
            }
        }
    },
}

INDEX_MAPPING = {
    "dynamic": "true",
    "properties": {
        "offer_id": {"type": "keyword"},
        "ast_path": {"type": "keyword", "ignore_above": 2048},
        "offer_title": {
            "type": "text",
            "analyzer": "offer_text",
            "fields": {"raw": {"type": "keyword", "ignore_above": 512}},
        },
        "offer_summary": {
            "type": "text",
            "analyzer": "offer_text",
            "fields": {"raw": {"type": "keyword", "ignore_above": 256}},
        },
        "offer_description": {"type": "text", "analyzer": "offer_text"},
        "offer_text": {"type": "text", "analyzer": "offer_text"},
        "searchable_text": {"type": "text", "analyzer": "offer_text"},
        "embedding": {
            "type": "dense_vector",
            "dims": EMBEDDING_DIMENSIONS,
            "index": True,
            "similarity": "cosine",
        },
        "primary_category": {"type": "keyword"},
        "secondary_category": {"type": "keyword"},
        "offer_types": {"type": "keyword"},
        "channel": {"type": "keyword"},
        "platform": {"type": "keyword"},
        "is_pan_india": {"type": "keyword"},
        "international": {"type": "keyword"},
        "card_tier": {"type": "keyword"},
        "card_network": {"type": "keyword"},
        "all_sbi_cards": {"type": "keyword"},
        "corporate_card_eligible": {"type": "keyword"},
        "brand_name": {
            "type": "text",
            "analyzer": "offer_text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "merchant_name": {
            "type": "text",
            "analyzer": "offer_text",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "brand_name_keyword": {"type": "keyword", "ignore_above": 256},
        "merchant_name_keyword": {"type": "keyword", "ignore_above": 256},
        "minimum_transaction_amount": {"type": "double"},
        "fuel_surcharge_waiver_percentage": {"type": "double"},
        "annual_fee_waiver_threshold": {"type": "double"},
        "start_date": {"type": "date", "format": "strict_date_optional_time||yyyy-MM-dd"},
        "end_date": {"type": "date", "format": "strict_date_optional_time||yyyy-MM-dd"},
        "discount": {
            "type": "object",
            "properties": {
                "discount_type": {"type": "keyword"},
                "discount_percentage": {"type": "double"},
                "discount_min_percentage": {"type": "double"},
                "discount_max_percentage": {"type": "double"},
                "discount_flat_amount": {"type": "double"},
                "discount_min_amount": {"type": "double"},
                "discount_max_amount": {"type": "double"},
            },
        },
        "cashback": {
            "type": "object",
            "properties": {
                "cashback_type": {"type": "keyword"},
                "cashback_percentage": {"type": "double"},
                "cashback_min_percentage": {"type": "double"},
                "cashback_max_percentage": {"type": "double"},
                "cashback_flat_amount": {"type": "double"},
                "cashback_min_amount": {"type": "double"},
                "cashback_max_amount": {"type": "double"},
            },
        },
        "emi_details": {
            "type": "object",
            "properties": {
                "zero_down_payment": {"type": "keyword"},
                "minimum_emi_tenure_months": {"type": "integer"},
                "maximum_emi_tenure_months": {"type": "integer"},
            },
        },
        "processing_fee": {
            "type": "object",
            "properties": {
                "processing_fee_amount": {"type": "double"},
                "processing_fee_percentage": {"type": "double"},
            },
        },
        "reward_points": {
            "type": "object",
            "properties": {
                "reward_multiplier": {"type": "double"},
            },
        },
        "eligible_cards": {"type": "keyword", "ignore_above": 256},
        "cities": {"type": "keyword", "ignore_above": 128},
        "cities_search": {"type": "text", "analyzer": "offer_text"},
        "eligible_cards_search": {"type": "text", "analyzer": "offer_text"},
        "brand_image": {"type": "keyword", "ignore_above": 1024},
        "offer_image": {"type": "keyword", "ignore_above": 1024},
    },
}


def build_standalone_index_body(settings: AppSettings) -> dict[str, Any]:
    body = {"settings": dict(INDEX_SETTINGS), "mappings": dict(INDEX_MAPPING)}
    dims = int(settings.embedding_dims or EMBEDDING_DIMENSIONS)
    body["mappings"]["properties"]["embedding"]["dims"] = dims
    return body


# ---------------------------------------------------------------------------
# Pydantic models (standalone_ingest.py)
# ---------------------------------------------------------------------------
class DiscountBlock(BaseModel):
    discount_type: str | None = None
    discount_percentage: float | None = None
    discount_min_percentage: float | None = None
    discount_max_percentage: float | None = None
    discount_flat_amount: float | None = None
    discount_min_amount: float | None = None
    discount_max_amount: float | None = None


class CashbackBlock(BaseModel):
    cashback_type: str | None = None
    cashback_percentage: float | None = None
    cashback_min_percentage: float | None = None
    cashback_max_percentage: float | None = None
    cashback_flat_amount: float | None = None
    cashback_min_amount: float | None = None
    cashback_max_amount: float | None = None


class EmiDetailsBlock(BaseModel):
    zero_down_payment: str | None = None
    minimum_emi_tenure_months: int | float | None = None
    maximum_emi_tenure_months: int | float | None = None


class ProcessingFeeBlock(BaseModel):
    processing_fee_amount: float | None = None
    processing_fee_percentage: float | None = None


class RewardPointsBlock(BaseModel):
    reward_multiplier: float | None = None


class OfferRecord(BaseModel):
    offer_title: str
    offer_description: str | None = None
    offer_summary: str | None = None
    primary_category: list[str] = Field(default_factory=list)
    secondary_category: list[str] = Field(default_factory=list)
    offer_types: list[str] = Field(default_factory=list)
    channel: str | None = None
    platform: list[str] | None = None
    is_pan_india: str | None = None
    international: str | None = None
    card_tier: list[str] | None = None
    card_network: str | None = None
    all_sbi_cards: str | None = None
    corporate_card_eligible: str | None = None
    brand_name: str | None = None
    merchant_name: str | None = None
    discount: DiscountBlock | None = None
    cashback: CashbackBlock | None = None
    emi_details: EmiDetailsBlock | None = None
    processing_fee: ProcessingFeeBlock | None = None
    reward_points: RewardPointsBlock | None = None
    fuel_surcharge_waiver_percentage: float | None = None
    annual_fee_waiver_threshold: float | None = None
    minimum_transaction_amount: float | None = None
    eligible_cards: list[str] = Field(default_factory=list)
    ast_path: str
    incoming_offer_id: str | None = Field(default=None, validation_alias="offer_id")
    offer_text: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    brand_image: str | None = None
    offer_image: str | None = None
    cities: list[str] = Field(default_factory=list)

    @staticmethod
    def _coerce_block(model: type[BaseModel], v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, model):
            return v
        if isinstance(v, dict):
            return model.model_validate(v)
        return v

    @field_validator("discount", mode="before")
    @classmethod
    def _discount(cls, v: Any) -> Any:
        return cls._coerce_block(DiscountBlock, v)

    @field_validator("cashback", mode="before")
    @classmethod
    def _cashback(cls, v: Any) -> Any:
        return cls._coerce_block(CashbackBlock, v)

    @field_validator("emi_details", mode="before")
    @classmethod
    def _emi(cls, v: Any) -> Any:
        return cls._coerce_block(EmiDetailsBlock, v)

    @field_validator("processing_fee", mode="before")
    @classmethod
    def _proc(cls, v: Any) -> Any:
        return cls._coerce_block(ProcessingFeeBlock, v)

    @field_validator("reward_points", mode="before")
    @classmethod
    def _reward(cls, v: Any) -> Any:
        return cls._coerce_block(RewardPointsBlock, v)

    @field_validator("platform", mode="before")
    @classmethod
    def _platform(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v

    model_config = {"extra": "ignore"}


class OfferRecordNormalized(OfferRecord):
    cities_normalized: list[str] = Field(default_factory=list)
    searchable_text: str = ""
    offer_id: str = ""
    start_date_parsed: date | None = None
    end_date_parsed: date | None = None


def normalize_cities(raw: list[str]) -> list[str]:
    out: list[str] = []
    for c in raw:
        if not c:
            continue
        s = str(c).strip()
        if "," in s:
            out.extend(x.strip() for x in s.split(",") if x.strip())
        elif s:
            out.append(s)
    return sorted(set(out))[:15000]


def stable_offer_id(ast_path: str) -> str:
    return hashlib.sha256(ast_path.encode("utf-8")).hexdigest()


def resolve_offer_id(offer: OfferRecord) -> str:
    raw = (offer.incoming_offer_id or "").strip()
    if raw:
        return raw
    return stable_offer_id(offer.ast_path)


def build_searchable_text(offer: OfferRecord) -> str:
    parts = [
        offer.offer_text or "",
        offer.offer_summary or "",
        offer.offer_description or "",
        offer.offer_title,
        offer.brand_name or "",
        offer.merchant_name or "",
        " ".join(offer.offer_types or []),
        " ".join(offer.primary_category or []),
        " ".join(offer.secondary_category or []),
    ]
    return "\n".join(p.strip() for p in parts if p and str(p).strip())


def _parse_iso(d: str | None) -> date | None:
    if not d:
        return None
    try:
        return date.fromisoformat(str(d)[:10])
    except ValueError:
        return None


def _emi_for_es(emi: Any) -> dict[str, Any] | None:
    if emi is None:
        return None
    d = emi.model_dump() if hasattr(emi, "model_dump") else dict(emi)
    for k in ("minimum_emi_tenure_months", "maximum_emi_tenure_months"):
        if d.get(k) is not None:
            try:
                d[k] = int(d[k])
            except (TypeError, ValueError):
                d[k] = None
    return d


def to_elasticsearch_document(offer: OfferRecordNormalized, embedding: list[float]) -> dict[str, Any]:
    src = offer.model_dump(
        exclude={
            "cities_normalized",
            "start_date_parsed",
            "end_date_parsed",
            "incoming_offer_id",
        }
    )
    src["cities"] = offer.cities_normalized or src.get("cities") or []
    cities_list = [str(c).strip() for c in src["cities"] if c and str(c).strip()]
    src["cities_search"] = " ".join(cities_list)
    ec_raw = src.get("eligible_cards") or []
    ec_list = [str(x).strip() for x in ec_raw if x and str(x).strip()]
    src["eligible_cards_search"] = " ".join(ec_list)
    src["searchable_text"] = offer.searchable_text
    src["embedding"] = embedding
    src["offer_id"] = offer.offer_id
    src["brand_name_keyword"] = (offer.brand_name or "").strip()
    src["merchant_name_keyword"] = (offer.merchant_name or "").strip()
    src["emi_details"] = _emi_for_es(offer.emi_details)
    return src


def _llm_block(parsed: dict[str, Any], llm: Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(llm, dict):
        return dict(llm)
    return {}


def _detail_plain(parsed: dict[str, Any]) -> str:
    parts: list[str] = []
    for section in parsed.get("detail_sections") or []:
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading") or "").strip()
        content = str(section.get("content") or "").strip()
        if heading:
            parts.append(heading)
        if content:
            parts.append(content)
    return "\n".join(parts).strip()


def build_offer_record_from_sources(
    parsed: dict[str, Any],
    *,
    llm: Mapping[str, Any] | None = None,
    page_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge parsed XML offer + optional LLM extraction into standalone OfferRecord input."""
    llm = _llm_block(parsed, llm)
    if page_meta:
        nested = page_meta.get("llm_enrichment")
        if isinstance(nested, dict):
            merged = dict(nested)
            merged.update({k: v for k, v in llm.items() if v is not None})
            llm = merged

    offer_id = str(parsed.get("offer_id") or "").strip()
    ast_path = str(parsed.get("source_url") or "").strip()
    if not ast_path and offer_id:
        ast_path = f"https://www.sbicard.com/en/personal/offer/{offer_id}"

    brand = str(parsed.get("brand_name") or llm.get("brand_name") or "").strip()
    offer_text = str(parsed.get("offer_text") or "").strip()
    discount_text = str(parsed.get("discount_text") or "").strip()

    offer_title = str(
        llm.get("offer_title")
        or llm.get("ai_title")
        or (f"{brand} - {offer_text}" if brand and offer_text else brand or offer_text or "SBI Card Offer")
    ).strip()

    detail_plain = _detail_plain(parsed)
    offer_description = str(
        llm.get("offer_description")
        or llm.get("ai_summary")
        or discount_text
        or detail_plain[:4000]
        or offer_text
    ).strip() or None

    offer_summary = str(
        llm.get("offer_summary")
        or llm.get("semantic_summary")
        or llm.get("ai_summary")
        or offer_description
        or ""
    ).strip() or None

    primary = llm.get("primary_category")
    if not primary:
        cat = str(parsed.get("category") or "").strip()
        primary = [cat] if cat else []
    elif isinstance(primary, str):
        primary = [primary]

    secondary = llm.get("secondary_category")
    if isinstance(secondary, str):
        secondary = [secondary]
    if not isinstance(secondary, list):
        secondary = secondary or []

    offer_types = llm.get("offer_types")
    if not offer_types:
        ot = str(parsed.get("offer_type") or "").strip()
        offer_types = [ot] if ot else []
    elif isinstance(offer_types, str):
        offer_types = [offer_types]

    channel = llm.get("channel") or llm.get("offer_channel")
    if not channel:
        channel = "online" if parsed.get("is_online") else "offline"

    platform = llm.get("platform")
    if isinstance(platform, str):
        platform = [platform]

    merchant = str(llm.get("merchant_name") or brand or "").strip() or None

    return {
        "offer_id": offer_id,
        "offer_title": offer_title,
        "offer_description": offer_description,
        "offer_summary": offer_summary,
        "primary_category": primary,
        "secondary_category": secondary,
        "offer_types": offer_types,
        "channel": channel,
        "platform": platform,
        "is_pan_india": llm.get("is_pan_india"),
        "international": llm.get("international"),
        "card_tier": llm.get("card_tier"),
        "card_network": llm.get("card_network"),
        "all_sbi_cards": llm.get("all_sbi_cards"),
        "corporate_card_eligible": llm.get("corporate_card_eligible"),
        "brand_name": brand or None,
        "merchant_name": merchant,
        "discount": llm.get("discount"),
        "cashback": llm.get("cashback"),
        "emi_details": llm.get("emi_details"),
        "processing_fee": llm.get("processing_fee"),
        "reward_points": llm.get("reward_points"),
        "fuel_surcharge_waiver_percentage": llm.get("fuel_surcharge_waiver_percentage"),
        "annual_fee_waiver_threshold": llm.get("annual_fee_waiver_threshold"),
        "minimum_transaction_amount": llm.get("minimum_transaction_amount"),
        "eligible_cards": parsed.get("eligible_cards") or [],
        "ast_path": ast_path,
        "offer_text": offer_text or None,
        "start_date": parsed.get("offer_start_date") or llm.get("start_date"),
        "end_date": parsed.get("offer_end_date") or llm.get("end_date"),
        "brand_image": parsed.get("revamp_image") or parsed.get("banner_image"),
        "offer_image": parsed.get("offer_image"),
        "cities": parsed.get("cities") or [],
    }


def validate_and_normalize_offers(raw_records: list[dict[str, Any]]) -> tuple[list[OfferRecordNormalized], list[str]]:
    valid: list[OfferRecord] = []
    errors: list[str] = []
    for i, row in enumerate(raw_records):
        try:
            valid.append(OfferRecord.model_validate(row))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"row {i} ({row.get('offer_id')}): {exc}")
    normalized: list[OfferRecordNormalized] = []
    for o in valid:
        cities_n = normalize_cities(o.cities)
        st = build_searchable_text(o)
        oid = resolve_offer_id(o)
        normalized.append(
            OfferRecordNormalized.model_validate(
                {
                    **o.model_dump(),
                    "cities_normalized": cities_n,
                    "searchable_text": st,
                    "offer_id": oid,
                    "start_date_parsed": _parse_iso(o.start_date),
                    "end_date_parsed": _parse_iso(o.end_date),
                }
            )
        )
    return normalized, errors


def build_records_for_pipeline(
    offers: list[dict[str, Any]],
    *,
    llm_by_offer_id: dict[str, dict[str, Any]] | None = None,
    llm_by_source_url: dict[str, dict[str, Any]] | None = None,
    changed_normalized_urls: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Build raw OfferRecord dicts; optionally filter to changed URLs only."""
    llm_by_offer_id = llm_by_offer_id or {}
    llm_by_source_url = llm_by_source_url or {}
    changed = changed_normalized_urls or set()
    out: list[dict[str, Any]] = []
    for parsed in offers:
        if not isinstance(parsed, dict):
            continue
        src = str(parsed.get("source_url") or "")
        norm = normalize_url(src)
        if changed and norm and norm not in changed:
            continue
        oid = str(parsed.get("offer_id") or "")
        llm = llm_by_offer_id.get(oid) or (llm_by_source_url.get(norm) if norm else None)
        out.append(build_offer_record_from_sources(parsed, llm=llm))
    return out


def url_hash_to_offer_id_map(offers: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for o in offers:
        if not isinstance(o, dict):
            continue
        oid = str(o.get("offer_id") or "").strip()
        norm = normalize_url(str(o.get("source_url") or ""))
        if oid and norm:
            mapping[build_doc_id(norm)] = oid
    return mapping


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        item = json.loads(text)
        if isinstance(item, dict):
            out.append(item)
    return out


def build_llm_lookup(llm_records: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_url: dict[str, dict[str, Any]] = {}
    for rec in llm_records:
        if not isinstance(rec, dict):
            continue
        oid = str(rec.get("offer_id") or "").strip()
        if oid:
            by_id[oid] = rec
        src = str(rec.get("source_url") or "").strip()
        norm = normalize_url(src)
        if norm:
            by_url[norm] = rec
    return by_id, by_url
