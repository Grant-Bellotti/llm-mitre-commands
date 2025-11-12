#!/usr/bin/env python3
"""
generate_placeholders.py

Reads /mnt/data/mitre_enterprise_techniques.json (MITRE techniques)
and outputs a JSONL file with one placeholder token per technique+platform.

Output file: mitre_placeholders.jsonl
Each line: {"mitre_id": "...", "name": "...", "platform": "...", "placeholder": "<COMMAND_T1003_linux>"}
"""
import json
from pathlib import Path

INPUT = "mitre_enterprise_techniques.json"
OUTPUT = "mitre_placeholders.jsonl"

def load_mitre(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_platform(p):
    if not p:
        return "unknown"
    
    returnList = []
    for i in p:
        returnList.append(i.lower().replace(" ", "_").replace("/", "_"))
    # make simple normalized label
    return returnList

def make_placeholder(mitre_id, platforms):
    pid = mitre_id or "UNKNOWN_ID"
    plat = normalize_platform(platforms)
    # safe deterministic placeholder
    return f"<COMMAND_{pid}_{plat}>"

def main():
    techniques = load_mitre(INPUT)
    out_path = Path(OUTPUT)
    if out_path.exists():
        out_path.unlink()

    with open(OUTPUT, "w", encoding="utf-8") as fh:
        for t in techniques:
            mitre_id = t.get("mitre_id") or t.get("id") or t.get("name")
            name = t.get("name","")
            platforms = t.get("platforms") or ["unknown"]
            # for p in platforms:
            record = {
                "mitre_id": mitre_id,
                "name": name,
                "platform": platforms,
                "placeholder": make_placeholder(mitre_id, platforms)
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote placeholders to {OUTPUT}")

if __name__ == "__main__":
    main()
