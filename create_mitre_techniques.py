import json
import requests

# URL to the MITRE attack data repo enterprise collection
MITRE_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"

OUTFILE = "mitre_enterprise_techniques.json"

# Get the MITRE ID if present
def getID(externalReference):
    if not externalReference:
        return None
    
    for ref in externalReference:
        # MITRE uses source_name "mitre-attack" and external_id for id
        if ref.get("source_name") in ("mitre-attack", "mitre"):
            if "external_id" in ref:
                return ref["external_id"]
            if "external_id" in ref.get("external_id", {}):
                return ref["external_id"]
            
    # Return first external_id found if no others found
    for ref in externalReference:
        if "external_id" in ref:
            return ref["external_id"]
        
    return None

# Extract all the data in the Mitre techniques
def extractTechniques(data):
    outData = []
    objects = data.get("objects", [])

    for obj in objects:
        # attack-pattern objects are techniques/sub-techniques
        if obj.get("type") != "attack-pattern":
            continue

        # data
        name = obj.get("name")
        description = obj.get("description") or ""
        platforms = obj.get("x_mitre_platforms") or ["unknown"]
        subtechnique = obj.get("x_mitre_is_subtechnique", False)
        externalID = getID(obj.get("external_references", []))

        # Limit the amount of techniques to only non subtechniques or techniques with specified platforms
        if (not subtechnique) or (platforms[0] in ["Linux", "Windows", "macOS"]):
            technique = {
                externalID : {
                    "mitre_id": externalID,
                    "name": name,
                    "description": description.strip(),
                    "is_subtechnique": bool(subtechnique),
                    "platforms": platforms[0]
                }
            }

            outData.append(technique)

    return outData

def main():
    print("Downloading STIX collection from MITRE attack-stix-data...")
    r = requests.get(MITRE_URL, timeout=60)
    r.raise_for_status()
    mitreData = r.json()

    print("Extracting techniques / sub-techniques...")
    techniques = extractTechniques(mitreData)
    print(f"Found {len(techniques)} 'attack-pattern' objects (techniques + sub-techniques).")

    print(f"Writing output to {OUTFILE}...")
    with open(OUTFILE, "w", encoding="utf-8") as f:
        json.dump(techniques, f, indent=2, ensure_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    main()
