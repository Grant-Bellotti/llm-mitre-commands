#!/usr/bin/env python3
"""
export_mitre_techniques.py

Downloads MITRE ATT&CK Enterprise STIX JSON and extracts all techniques / sub-techniques
into a simple JSON file: mitre_enterprise_techniques.json

Requires: requests
pip install requests
"""

import json
import requests
from urllib.parse import urljoin

# Raw URL to the MITRE attack-stix-data repo enterprise collection (STIX 2.1)
STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"

OUTFILE = "mitre_enterprise_techniques.json"

def canonical_external_id(ext_refs):
    """Return the MITRE external_id (e.g., 'T1087' or 'T1548.001') if present"""
    if not ext_refs:
        return None
    for ref in ext_refs:
        # MITRE uses source_name "mitre-attack" and external_id for canonical id
        if ref.get("source_name") in ("mitre-attack", "mitre"):
            if "external_id" in ref:
                return ref["external_id"]
            if "external_id" in ref.get("external_id", {}):
                return ref["external_id"]
    # fallback: return first external_id found
    for ref in ext_refs:
        if "external_id" in ref:
            return ref["external_id"]
    return None

def extract_techniques(stix_bundle):
    """
    stix_bundle: parsed JSON object from the MITRE collection
    returns: list of technique dicts
    """
    out = []
    objects = stix_bundle.get("objects", [])
    for obj in objects:
        # attack-pattern objects are techniques/sub-techniques
        if obj.get("type") != "attack-pattern":
            continue

        # Common STIX properties
        name = obj.get("name")
        description = obj.get("description") or ""
        x_mitre_platforms = obj.get("x_mitre_platforms") or ["unknown"]
        x_mitre_detection = obj.get("x_mitre_detection") or ""
        x_mitre_data_sources = obj.get("x_mitre_data_sources") or []
        x_mitre_permissions_required = obj.get("x_mitre_permissions_required") or []
        x_mitre_is_subtechnique = obj.get("x_mitre_is_subtechnique", False)
        x_mitre_tactic = obj.get("x_mitre_tactic") or obj.get("kill_chain_phases") or []
        x_mitre_contributors = obj.get("x_mitre_contributors") or []

        external_id = canonical_external_id(obj.get("external_references", []))
        shortname = obj.get("x_mitre_shortname") or None

        # references: collect urls in external_references
        references = []
        for r in obj.get("external_references", []):
            if "url" in r:
                references.append(r["url"])
            elif "source_name" in r:
                references.append(r["source_name"])

        # If it's a sub-technique, try to find parent when present as 'x_mitre_platforms' —
        # (Note: full parent linking is available via Relationship objects in STIX; we skip full graph traversal
        # to keep output self-contained. See USAGE if you want parent-child resolved.)
        technique = {
            external_id : {
                "mitre_id": external_id,
                "name": name,
                "description": description.strip(),
                "is_subtechnique": bool(x_mitre_is_subtechnique),
                "platforms": x_mitre_platforms[0]
            }
        }

        out.append(technique)

    return out

def main():
    print("Downloading STIX collection from MITRE attack-stix-data...")
    r = requests.get(STIX_URL, timeout=60)
    r.raise_for_status()
    stix = r.json()

    print("Extracting techniques / sub-techniques...")
    techniques = extract_techniques(stix)
    print(f"Found {len(techniques)} 'attack-pattern' objects (techniques + sub-techniques).")

    print(f"Writing output to {OUTFILE} ...")
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(techniques, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Output file: {OUTFILE}")
    print("Note: For full parent <-> sub-technique linking, parse relationship objects in the STIX bundle or use MITRE CTI/attack-stix-data tools.")

if __name__ == "__main__":
    main()
