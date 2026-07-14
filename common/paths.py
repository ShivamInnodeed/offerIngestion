from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
INGESTION_DIR = ROOT_DIR.parent / "ingestion"
OUTPUT_DIR = ROOT_DIR / "output"
CONFIG_DIR = ROOT_DIR / "config"
XML_DATA_DIR = ROOT_DIR / "all_xml_data"
