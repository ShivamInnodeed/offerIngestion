"""LLM enrichment for SBI Card offer records."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from offer_llm_json_parse import parse_llm_json_from_response
from offer_llm_system_prompt import OFFER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class LlmConfig:
    api_url: str
    model: str
    timeout_s: int
    temperature: float
    auth_header: str | None
    auth_value: str | None
    max_retries: int
    retry_delay_s: float


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def load_config() -> LlmConfig:
    api_url = _env("LLM_API_URL", "http://localhost:8000/v1/chat/completions") or ""
    model = _env("LLM_MODEL", "gpt-oss-20b") or ""
    timeout_s = int(_env("LLM_TIMEOUT", "180") or "180")
    temperature = float(_env("LLM_TEMPERATURE", "0") or "0")
    max_retries = max(0, int(_env("LLM_MAX_RETRIES", "3") or "3"))
    retry_delay_s = max(0.0, float(_env("LLM_RETRY_DELAY_SECONDS", "2") or "2"))

    bearer = _env("LLM_BEARER_TOKEN")
    auth_header = _env("LLM_AUTH_HEADER")
    auth_value = _env("LLM_AUTH_VALUE")
    if bearer and not (auth_header or auth_value):
        auth_header = "Authorization"
        auth_value = f"Bearer {bearer}"

    return LlmConfig(
        api_url=api_url,
        model=model,
        timeout_s=timeout_s,
        temperature=temperature,
        auth_header=auth_header,
        auth_value=auth_value,
        max_retries=max_retries,
        retry_delay_s=retry_delay_s,
    )


def normalize_llm_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Map extraction-schema LLM output to pipeline-compatible fields for ES/search."""
    out = dict(rec)

    offer_title = rec.get("offer_title")
    offer_desc = rec.get("offer_description")
    offer_summary = rec.get("offer_summary")

    if offer_title and not out.get("ai_title"):
        out["ai_title"] = offer_title
    if offer_summary and not out.get("ai_summary"):
        out["ai_summary"] = offer_summary
    elif offer_desc and not out.get("ai_summary"):
        out["ai_summary"] = offer_desc
    if offer_summary and not out.get("semantic_summary"):
        out["semantic_summary"] = offer_summary

    primary = rec.get("primary_category")
    if isinstance(primary, list) and primary and not out.get("offer_category_refined"):
        out["offer_category_refined"] = str(primary[0])

    offer_types = rec.get("offer_types")
    if isinstance(offer_types, list) and offer_types and not out.get("offer_benefit_type"):
        out["offer_benefit_type"] = str(offer_types[0])

    channel = rec.get("channel")
    if channel and not out.get("offer_channel"):
        out["offer_channel"] = str(channel)

    secondary = rec.get("secondary_category")
    if isinstance(secondary, list) and secondary and not out.get("merchant_category"):
        out["merchant_category"] = str(secondary[0])

    if not out.get("keywords"):
        kw: list[str] = []
        for name in (rec.get("brand_name"), rec.get("merchant_name")):
            if name:
                kw.append(str(name).strip().lower())
        if isinstance(primary, list):
            kw.extend(str(x).strip().lower() for x in primary[:3] if x)
        if isinstance(offer_types, list):
            kw.extend(str(x).strip().lower() for x in offer_types[:2] if x)
        out["keywords"] = list(dict.fromkeys(k for k in kw if k))[:14]

    return out


def apply_llm_record_to_page(page: dict[str, Any], rec: dict[str, Any]) -> None:
    """Merge normalized LLM extraction into a crawl page record."""
    title = rec.get("ai_title") or rec.get("offer_title")
    summary = (
        rec.get("ai_summary")
        or rec.get("offer_summary")
        or rec.get("offer_description")
    )
    if title:
        page["title"] = title
    if summary:
        page["description"] = summary

    if isinstance(rec.get("keywords"), list):
        page["keywords"] = ",".join(str(k) for k in rec["keywords"] if k)

    meta = dict(page.get("metadata") or {})
    for k in (
        "ai_title",
        "ai_summary",
        "semantic_summary",
        "keywords",
        "offer_category_refined",
        "offer_benefit_type",
        "offer_channel",
        "merchant_category",
        "target_audience",
        "offer_title",
        "offer_description",
        "offer_summary",
        "customer_blurb",
        "primary_category",
        "secondary_category",
        "offer_types",
        "channel",
        "platform",
        "is_pan_india",
        "international",
    ):
        if k in rec:
            meta[k] = rec[k]

    meta["llm_enrichment"] = rec
    if title:
        meta["title"] = title
    if summary:
        meta["description"] = summary
    page["metadata"] = meta


def _call_llm_once(config: LlmConfig, *, offer: dict[str, Any]) -> dict[str, Any]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.auth_header and config.auth_value:
        headers[config.auth_header] = config.auth_value

    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": [
            {"role": "system", "content": OFFER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(offer, ensure_ascii=False)},
        ],
    }
    resp = requests.post(config.api_url, json=payload, headers=headers, timeout=config.timeout_s)
    resp.raise_for_status()
    return normalize_llm_record(parse_llm_json_from_response(resp.json()))


def call_llm_with_retries(
    config: LlmConfig,
    *,
    offer: dict[str, Any],
    offer_id: str,
    offer_index: int,
    total_offers: int,
) -> dict[str, Any]:
    attempts = config.max_retries + 1
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _call_llm_once(config, offer=offer)
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            logger.warning(
                "LLM attempt %s/%s failed for offer %s/%s (%s): %s — retrying in %.1fs",
                attempt,
                attempts,
                offer_index + 1,
                total_offers,
                offer_id or "<missing>",
                exc,
                config.retry_delay_s,
            )
            if config.retry_delay_s > 0:
                time.sleep(config.retry_delay_s)
    assert last_exc is not None
    raise last_exc


def enrich_offers(
    offers: list[dict[str, Any]],
    *,
    config: LlmConfig,
    max_offers: int | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    limit = max_offers if (isinstance(max_offers, int) and max_offers > 0) else None
    failed = 0

    for idx, offer in enumerate(offers):
        if limit is not None and idx >= limit:
            break

        offer_id = str(offer.get("offer_id") or "").strip()
        logger.info("Enriching offer %s/%s %s", idx + 1, len(offers), offer_id or "<missing>")
        try:
            enriched = call_llm_with_retries(
                config,
                offer=offer,
                offer_id=offer_id,
                offer_index=idx,
                total_offers=len(offers),
            )
        except Exception as exc:
            failed += 1
            logger.warning(
                "LLM enrich failed for offer %s/%s (%s) after %s attempt(s): %s",
                idx + 1,
                len(offers),
                offer_id,
                config.max_retries + 1,
                exc,
            )
            continue

        enriched["offer_id"] = offer_id
        enriched["source_url"] = offer.get("source_url", "")
        enriched["offer_index"] = idx
        results.append(enriched)

    if failed:
        logger.info("LLM enrich finished: %s success, %s failed", len(results), failed)
    return results


def read_offers(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [o for o in data if isinstance(o, dict)]
    raise ValueError("Input must be a JSON array of offer objects.")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich SBI Card offers via LLM.")
    parser.add_argument("--input", required=True, help="Input JSON file (list of offers).")
    parser.add_argument("--output", required=True, help="Output JSON file (enriched records).")
    parser.add_argument("--max-offers", type=int, default=0, help="Limit offers (0 = no limit).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    offers = read_offers(Path(args.input))
    config = load_config()
    results = enrich_offers(offers, config=config, max_offers=args.max_offers or None)
    write_json(Path(args.output), results)
    logger.info("Wrote %s enriched record(s) to %s", len(results), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
