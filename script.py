from __future__ import annotations

import os
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# Disable SSL verification (same as curl -k)
ssl._create_default_https_context = ssl._create_unverified_context


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value and value.strip() else default


def main() -> int:
    # Where to write the generated XML (must be a writable mount in container/server)
    out_dir = Path(_env("OFFER_XML_OUT_DIR", "/app/offerIngestion/all_xml_data"))
    out_path = Path(_env("OFFER_XML_OUT_PATH", str(out_dir / "all_offer_data1.xml")))
    out_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.today().strftime("%Y-%m-%d")
    url = (
        "https://172.16.23.189/lscs/v1/document$?"
        f"q=(TeamSite/Templating/DCR/Type:=%22product/offers%22)AND%20"
        f"(sbi.offers.enddate:%3E={today}%20AND%20sbi.offers.startdate:%3C={today})AND%20"
        "(sbi.offers.iss2s:=N)%20AND%20(sbi.offers.pagetype:=Personal)%20AND%20"
        "(sbi.offers.pre.login.flag:=Y)"
        "&project=/default/main/sbi-card/en_IN&start=0&max=1000"
    )

    with urllib.request.urlopen(url) as response:
        metadata_xml = response.read().decode("utf-8")

    root = ET.fromstring(metadata_xml)
    ns = {"ns": "http://www.interwoven.com/schema/iwrr"}

    with out_path.open("w", encoding="utf-8") as f:
        f.write("<all_offers>\n")

        for doc in root.findall(".//ns:document", ns):
            doc_id = doc.attrib.get("id")
            f.write("<offer>\n")

            # Metadata block
            f.write("<metadata>\n")
            f.write(ET.tostring(doc, encoding="unicode"))
            f.write("\n</metadata>\n")

            full_url = (
                "https://lscsprod.sbic.sbicard.com/lscs/v1/document/id/"
                f"{doc_id}?project=/default/main/sbi-card/en_IN"
            )
            try:
                with urllib.request.urlopen(full_url) as resp:
                    full_xml = resp.read().decode("utf-8")
                    if full_xml.startswith("<?xml"):
                        full_xml = full_xml.split("?>", 1)[1]
                    f.write(full_xml.strip() + "\n")
            except Exception:
                f.write(f"<error>Failed to fetch {doc_id}</error>\n")

            f.write("</offer>\n")

        f.write("</all_offers>")

    print(f"Wrote offer XML to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())