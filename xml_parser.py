"""Parse SBI Card offer XML feed into structured JSON records."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

NS = {"ns0": "http://www.interwoven.com/schema/iwrr"}


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return (el.text or "").strip()


def _field_map(metadata_el: ET.Element) -> dict[str, str]:
    """Extract ns0:field name→value mapping from CMS metadata block."""
    fields: dict[str, str] = {}
    for doc in metadata_el.iter(f"{{{NS['ns0']}}}document"):
        for field in doc.iter(f"{{{NS['ns0']}}}field"):
            name = field.attrib.get("name", "")
            value = (field.text or "").strip()
            if name:
                fields[name] = value
    return fields


def _split_delimited(value: str) -> list[str]:
    """Split on `;` or `,` — the XML uses both depending on offer type."""
    if not value:
        return []
    if ";" in value:
        items = value.split(";")
    else:
        items = value.split(",")
    return [v.strip() for v in items if v.strip()]


def _clean_html(raw: str) -> str:
    """Strip HTML tags and decode entities to produce plain text."""
    if not raw:
        return ""
    decoded = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", decoded)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_offer_details(root_el: ET.Element) -> list[dict[str, str]]:
    """Extract offer detail sections (Summary, Steps, T&C, etc.)."""
    sections: list[dict[str, str]] = []
    for container in root_el.iter("offer_details_container"):
        tab_title = _text(container.find("tab_title"))
        for oc in container.findall("offer_container"):
            heading = _text(oc.find("heading"))
            raw_desc = _text(oc.find("description"))
            clean_desc = _clean_html(raw_desc)
            if heading or clean_desc:
                sections.append({
                    "tab": tab_title,
                    "heading": heading,
                    "content": clean_desc,
                })
    return sections


def _compute_offer_hash(offer: dict[str, Any]) -> str:
    """Deterministic hash of offer content for change detection."""
    parts = [
        str(offer.get("offer_id") or ""),
        str(offer.get("brand_name") or ""),
        str(offer.get("offer_text") or ""),
        str(offer.get("discount_text") or ""),
        str(offer.get("offer_start_date") or ""),
        str(offer.get("offer_end_date") or ""),
        str(offer.get("category") or ""),
        str(offer.get("offer_type") or ""),
        str(offer.get("priority") or ""),
        json.dumps(offer.get("detail_sections") or [], ensure_ascii=False, sort_keys=True),
        ";".join(sorted(offer.get("eligible_cards") or [])),
    ]
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def parse_single_offer(offer_el: ET.Element) -> dict[str, Any] | None:
    """Parse one <offer> element into a structured dict."""
    metadata_el = offer_el.find("metadata")
    root_el = offer_el.find("root")
    if root_el is None:
        return None

    fields = _field_map(metadata_el) if metadata_el is not None else {}

    cms_doc_id = ""
    if metadata_el is not None:
        for doc_el in metadata_el.iter(f"{{{NS['ns0']}}}document"):
            cms_doc_id = doc_el.attrib.get("id", "")
            break

    oc = root_el.find("offer_container")
    if oc is None:
        return None

    offer_id = _text(oc.find("offer_id")) or fields.get("sbi.offers.offerid", "")
    if not offer_id:
        return None

    brand_name = ""
    revamp = root_el.find("offers_revamp")
    if revamp is not None:
        brand_name = _text(revamp.find("brand_name"))
    if not brand_name:
        brand_name = fields.get("sbi.offers.brand.name", "")

    category = _text(oc.find("category")) or fields.get("sbi.offers.category", "")
    offer_type_raw = _text(oc.find("type")) or fields.get("sbi.offers.type", "")
    offer_type = offer_type_raw.strip().strip(",").strip()
    if not offer_type:
        dcr_path = fields.get("AreaRelativePath", "").lower()
        if "convert-to-emi" in dcr_path:
            offer_type = "Convert To EMI"
        elif "shopping" in dcr_path or "dept-stores" in category.lower():
            offer_type = "Shopping"
        elif "travel" in dcr_path or "travel" in category.lower():
            offer_type = "Travel"
        elif "dining" in dcr_path or "dining" in category.lower():
            offer_type = "Dining"
        elif "fuel" in dcr_path or "fuel" in category.lower():
            offer_type = "Fuel"
        elif "entertainment" in dcr_path or "entertainment" in category.lower():
            offer_type = "Entertainment"
    pagetype = _text(oc.find("pagetype")) or fields.get("sbi.offers.pagetype", "Personal")
    priority = _text(oc.find("priority")) or fields.get("sbi.offers.priority", "")
    is_online = (_text(oc.find("is_online")) or fields.get("sbi.offers.isonline", "")).upper() == "Y"
    is_s2s = (_text(oc.find("is_s2s")) or fields.get("sbi.offers.iss2s", "")).upper() == "Y"
    is_banner = (_text(oc.find("is_banner")) or fields.get("sbi.offers.isbanner", "")).upper() == "Y"

    offer_start_date = _text(oc.find("offer_start_date")) or fields.get("sbi.offers.startdate", "")
    offer_end_date = _text(oc.find("offer_end_date")) or fields.get("sbi.offers.enddate", "")

    cards_raw = _text(oc.find("card")) or fields.get("sbi.offers.card", "")
    eligible_cards = _split_delimited(cards_raw)

    cities_raw = _text(oc.find("city")) or fields.get("sbi.offers.city", "")
    cities = _split_delimited(cities_raw)
    all_city = (_text(oc.find("all_city")) or "").lower() == "yes"

    offer_text = ""
    non_s2s = root_el.find("non_s2s_offer_container")
    if non_s2s is not None:
        offer_text = _text(non_s2s.find("offer_text"))
    if not offer_text:
        offer_text = fields.get("sbi.offers.offertext", "")

    offer_image = ""
    offer_image_alt = ""
    if non_s2s is not None:
        img_cont = non_s2s.find("offer_image_container")
        if img_cont is not None:
            offer_image = _text(img_cont.find("image_url"))
            offer_image_alt = _text(img_cont.find("image_alt_text"))
    if not offer_image:
        offer_image = fields.get("sbi.offers.offerimage", "")
        offer_image_alt = fields.get("sbi.offers.offerimage.alttext", "")

    discount_text = ""
    if revamp is not None:
        discount_text = _text(revamp.find("discount_text"))
    if not discount_text:
        discount_text = fields.get("sbi.offers.discount.text", "")

    revamp_image = ""
    revamp_image_alt = ""
    if revamp is not None:
        img_c = revamp.find("offer_image_container_f")
        if img_c is not None:
            revamp_image = _text(img_c.find("image_url"))
            revamp_image_alt = _text(img_c.find("image_alt_text"))

    banner_image = ""
    banner_el = root_el.find("offer_banner_container")
    if banner_el is not None:
        banner_image = _text(banner_el.find("image_url"))

    countdown = (_text(oc.find("countdown")) or fields.get("sbi.offers.countdown", "")).upper() == "Y"
    countdown_start = _text(oc.find("countdown_start_date")) or fields.get("sbi.offers.countdownstartdate", "")

    category_emi = _text(oc.find("category_emi"))
    pre_login_flag = (_text(oc.find("pre_login_flag")) or fields.get("sbi.offers.pre.login.flag", "")).upper() in ("Y", "YES")

    detail_sections = _parse_offer_details(root_el)

    offer: dict[str, Any] = {
        "offer_id": offer_id,
        "brand_name": brand_name,
        "category": category,
        "category_emi": category_emi or None,
        "offer_type": offer_type,
        "pagetype": pagetype,
        "priority": int(priority) if priority.isdigit() else 0,
        "is_online": is_online,
        "is_s2s": is_s2s,
        "is_banner": is_banner,
        "pre_login": pre_login_flag,
        "offer_start_date": offer_start_date or None,
        "offer_end_date": offer_end_date or None,
        "eligible_cards": eligible_cards,
        "eligible_cards_count": len(eligible_cards),
        "cities": cities,
        "cities_count": len(cities),
        "all_city": all_city,
        "offer_text": offer_text,
        "offer_image": offer_image,
        "offer_image_alt": offer_image_alt,
        "discount_text": discount_text,
        "revamp_image": revamp_image,
        "revamp_image_alt": revamp_image_alt,
        "banner_image": banner_image,
        "countdown": countdown,
        "countdown_start_date": countdown_start or None,
        "detail_sections": detail_sections,
        "cms_doc_id": cms_doc_id or None,
        "source_url": f"https://www.sbicard.com/en/personal/offer/{offer_id}",
    }
    offer["content_hash"] = _compute_offer_hash(offer)
    return offer


def parse_xml_file(xml_path: Path) -> list[dict[str, Any]]:
    """Parse the entire XML file and return a list of offer dicts."""
    logger.info("Parsing XML file: %s", xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    offers: list[dict[str, Any]] = []
    for offer_el in root.iter("offer"):
        try:
            parsed = parse_single_offer(offer_el)
            if parsed:
                offers.append(parsed)
        except Exception:
            logger.exception("Failed to parse an offer element, skipping")

    logger.info("Parsed %s offers from XML", len(offers))
    return offers


def parse_xml_string(xml_string: str) -> list[dict[str, Any]]:
    """Parse XML string (e.g. from API response) into offer dicts."""
    root = ET.fromstring(xml_string)
    offers: list[dict[str, Any]] = []
    for offer_el in root.iter("offer"):
        try:
            parsed = parse_single_offer(offer_el)
            if parsed:
                offers.append(parsed)
        except Exception:
            logger.exception("Failed to parse an offer element, skipping")
    return offers


def build_offer_markdown(offer: dict[str, Any]) -> str:
    """Build a human-readable markdown representation of an offer for embedding."""
    parts: list[str] = []

    brand = offer.get("brand_name") or ""
    offer_text = offer.get("offer_text") or ""
    discount = offer.get("discount_text") or ""

    title = f"{brand} - {offer_text}" if brand and offer_text else (brand or offer_text or "SBI Card Offer")
    parts.append(f"# {title}")

    if discount:
        parts.append(f"\n**{discount}**")

    parts.append(f"\n- **Category:** {offer.get('category', 'N/A')}")
    parts.append(f"- **Type:** {offer.get('offer_type', 'N/A')}")
    if offer.get("offer_start_date"):
        parts.append(f"- **Valid from:** {offer['offer_start_date']}")
    if offer.get("offer_end_date"):
        parts.append(f"- **Valid until:** {offer['offer_end_date']}")
    parts.append(f"- **Online:** {'Yes' if offer.get('is_online') else 'No'}")
    if offer.get("all_city"):
        parts.append("- **Available:** All cities")
    elif offer.get("cities_count"):
        parts.append(f"- **Available in:** {offer['cities_count']} cities")
    parts.append(f"- **Eligible cards:** {offer.get('eligible_cards_count', 0)} card types")

    for section in (offer.get("detail_sections") or []):
        heading = section.get("heading") or section.get("tab") or ""
        content = section.get("content") or ""
        if heading:
            parts.append(f"\n## {heading}")
        if content:
            parts.append(content)

    return "\n".join(parts).strip()


def offers_to_crawl_records(offers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert parsed offers into crawl-page-compatible records for the shared audit pipeline."""
    records: list[dict[str, Any]] = []
    for offer in offers:
        markdown = build_offer_markdown(offer)
        title = offer.get("brand_name") or "SBI Offer"
        offer_text_val = offer.get("offer_text") or ""
        if offer_text_val:
            title = f"{title} | {offer_text_val}"

        record: dict[str, Any] = {
            "source_url": offer.get("source_url", ""),
            "title": title,
            "description": offer_text_val or (offer.get("discount_text") or ""),
            "keywords": ",".join([
                offer.get("brand_name") or "",
                offer.get("category") or "",
                offer.get("offer_type") or "",
            ]),
            "category": "Offers",
            "sub_category": offer.get("category"),
            "path": f"/en/personal/offer/{offer.get('offer_id', '')}",
            "depth": 0,
            "success": True,
            "status_code": 200,
            "markdown": markdown,
            "metadata": {
                "source_url": offer.get("source_url", ""),
                "title": title,
                "description": offer_text_val,
                "keywords": offer.get("category", ""),
                "offer_id": offer.get("offer_id"),
                "brand_name": offer.get("brand_name"),
                "offer_type": offer.get("offer_type"),
                "offer_category": offer.get("category"),
                "offer_start_date": offer.get("offer_start_date"),
                "offer_end_date": offer.get("offer_end_date"),
                "is_online": offer.get("is_online"),
                "priority": offer.get("priority"),
                "eligible_cards_count": offer.get("eligible_cards_count"),
                "cities_count": offer.get("cities_count"),
                "all_city": offer.get("all_city"),
            },
            "_offer_data": offer,
        }
        records.append(record)
    return records


def write_offers_json(offers: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(offers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s offers to %s", len(offers), output_path)
