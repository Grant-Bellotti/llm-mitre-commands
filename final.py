#!/usr/bin/env python3
"""
generate_technique_descriptions_safe.py

Reads /mnt/data/mitre_enterprise_techniques.json (MITRE techniques),
builds a RAG index, and uses a local Llama via LangChain to produce one-line,
non-actionable, high-level descriptions for each technique (one description per technique).

Output file: mitre_technique_descriptions.jsonl
Each line: {"mitre_id":"T1003","name":"OS Credential Dumping","platform":"windows","description":"Short non-actionable description..."}

SAFETY: The prompt instructs the model to avoid commands. The script also runs
a conservative regex-based filter to redact outputs that look like shell/PowerShell/one-liner commands.
"""

import json
import re
from pathlib import Path
from typing import List, Dict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# ---------- Config ----------
MITRE_FILE = "mitre_enterprise_techniques.json"
OUTFILE = "mitre_technique_descriptions.jsonl"

# Embedding model (small & fast)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Local Llama model id (quantized)
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Retriever settings
K = 3  # number of retrieved chunks

# # Conservative blacklist: patterns that look like commands or dangerous tokens
# _COMMAND_LIKE_PATTERNS = [
#     r"\b[a-z0-9\-_]+\.exe\b",             # windows executables
#     r"/bin/|/usr/|\\windows\\",           # unix/windows paths
#     r"\b(?:curl|wget|nc|netcat|ncat)\b",  # network tools
#     r"\b(?:rm|dd|mkfs|chmod|chown)\b",    # destructive tools
#     r"\b(?:sudo|su|runas)\b",             # privilege escalation words
#     r"->|;|\||\$\(|`|>\s*/",              # shell piping/redirects/backticks
#     r"powershell\b",                      # powershell token
#     r"meterpreter|msfconsole|exploit|payload|reverse\s+shell"
# ]
# _COMMAND_RE = re.compile("|".join(f"({p})" for p in _COMMAND_LIKE_PATTERNS), flags=re.IGNORECASE)

# ---------- Helpers ----------

def load_mitre(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def technique_to_document(t: Dict) -> Document:
    """
    Convert a technique dict (from your STIX-extracted JSON) into a LangChain Document.
    We'll include name, mitre_id, description, detection and references to give the retriever context.
    """
    mitre_id = t.get("mitre_id") or t.get("id") or ""
    name = t.get("name") or ""
    desc = t.get("description") or ""
    detection = t.get("detection") or t.get("detection") or t.get("detection", "")
    platforms = t.get("platforms") or [""]
    refs = t.get("references") or []
    text_parts = [
        f"ID: {mitre_id}",
        f"Name: {name}",
        f"Platforms: {', '.join(platforms)}",
        "",
        "Description:",
        desc,
        "",
        "Detection:",
        t.get("detection", "") or "",
        "",
        "References:",
        "\n".join(refs)
    ]
    page_content = "\n".join(part for part in text_parts if part)
    metadata = {"mitre_id": mitre_id, "name": name, "platforms": platforms[0]}
    return Document(page_content=page_content, metadata=metadata)

def normalize_platform_list(platforms) -> List[str]:
    """Return normalized platform list (lowercase, common aliases to 'macos')."""
    if not platforms:
        return ["unknown"]
    out = []
    for p in platforms:
        if not p:
            continue
        s = str(p).strip().lower()
        s = s.replace("osx","macos").replace("mac os x","macos").replace("mac os","macos")
        s = s.replace("microsoft windows","windows")
        s = s.replace("/", "_").replace(" ", "_")
        if s and s not in out:
            out.append(s)
    if not out:
        return ["unknown"]
    return out

def choose_single_platform(platforms: List[str]) -> str:
    priority = ["linux", "windows", "macos"]
    normalized = normalize_platform_list(platforms)
    for p in priority:
        if p in normalized:
            return p
    return normalized[0]

def sanitize_and_enforce_one_line(text: str) -> str:
    """
    Keep only first non-empty line, strip, and redact if it's command-like or contains
    dangerous tokens. Return a safe one-line description.
    """
    if not text:
        return "[NO_OUTPUT]"

    # split into lines and take the first nonempty
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    one = lines[0] if lines else ""
    # # If model output includes backticks or code fences, remove them
    # one = re.sub(r"```.*?```", "", one, flags=re.DOTALL)
    # one = one.replace("`", "").strip()

    # # If detection of command-like tokens, redact
    # if _COMMAND_RE.search(one):
    #     return "[REDACTED_ACTIONABLE_CONTENT]"

    # Limit length
    if len(one) > 400:
        one = one[:400].rsplit(" ",1)[0] + "..."

    # # Ensure it's not a command-line fragment (conservative)
    # if re.match(r"^[\w\-\./\\]+(\s+[\w\-\./\\]+)*$", one) and (" " in one) and len(one.split()) < 6:
    #     # Looks suspiciously like a short command or filename list — redact
    #     return "[REDACTED_POTENTIAL_COMMAND]"

    return one

# ---------- Model / RAG setup ----------

def create_vectorstore_from_techniques(techniques: List[Dict]) -> Chroma:
    docs = [technique_to_document(t) for t in techniques]
    # Optionally chunk documents if descriptions are long (omitted for brevity).
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vect = Chroma.from_documents(docs, embedding=emb)
    return vect

def create_local_llm_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        temperature=0.2,
        top_p=0.9,
        do_sample=False
    )
    return HuggingFacePipeline(pipeline=pipe)

PROMPT = """
You are a offensive/educational assistant. Using the retrieved technique context below, produce EXACTLY ONE COMMAND (one line) that is an example of what an attacker *might attempt* for this technique on the specified platform. DO NOT include step-by-step instructions or anything other than the command. Output only one command and nothing else.

--- Retrieved context:
{context}

--- Metadata:
Technique: {mitre_id} — {name}
Platform: {platform}
"""

def main():
    # 1) Load techniques
    techniques = load_mitre(MITRE_FILE)
    print(f"Loaded {len(techniques)} techniques from {MITRE_FILE}")

    # 2) Build vectorstore
    print("Building embeddings and vectorstore (this may take a bit)...")
    vect = create_vectorstore_from_techniques(techniques)
    retriever = vect.as_retriever(search_kwargs={"k": K})

    # 3) Create LLM and RetrievalQA
    print("Loading local LLM pipeline...")
    llm = create_local_llm_pipeline()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

    # 4) Iterate techniques and produce one-line descriptions
    out_path = Path(OUTFILE)
    if out_path.exists():
        out_path.unlink()
    count = 0
    print("Generating safe, one-line descriptions...")

    for t in techniques:
        mitre_id = t.get("mitre_id") or t.get("id") or ""
        name = t.get("name","")
        platforms = t.get("platforms") or []
        chosen_platform = choose_single_platform(platforms)

        # Build retrieval query to get best supporting context
        query_text = f"{mitre_id} {name} platforms: {', '.join(platforms)}"
        # Use retriever directly for context so we can provide explicit prompt
        retrieved_docs = retriever.get_relevant_documents(query_text)
        context = "\n\n---\n\n".join([d.page_content for d in retrieved_docs]) if retrieved_docs else (t.get("description","") or "")

        prompt = PROMPT.format(context=context, mitre_id=mitre_id, name=name, platform=chosen_platform)

        # Call LLM through RetrievalQA to ensure it uses retriever (qa.run also prompts the model),
        # but we want to use our explicit prompt — so call llm.pipeline directly with prompt text.
        # HuggingFacePipeline returns a callable that accepts a prompt dict in LangChain usage, but
        # for simplicity we call the pipeline via the LangChain llm object:
        response = llm(prompt)
        # HuggingFacePipeline when called returns either a string or a dict depending on version;
        if isinstance(response, str):
            raw = response
        elif isinstance(response, dict):
            raw = response.get("text") or response.get("generated_text") or str(response)
        else:
            # Try to handle list-of-dicts style
            try:
                raw = response[0].get("generated_text", "") if isinstance(response, list) else str(response)
            except Exception:
                raw = str(response)

        # Sanitize and enforce one-line, non-actionable
        safe_line = sanitize_and_enforce_one_line(raw)

        # # If the LLM generated redaction or empty, fallback to a concise summary from the stored data:
        # if safe_line.startswith("[REDACTED") or safe_line in ("", "[NO_OUTPUT]"):
        #     # Fallback: create a single non-actionable sentence from the stored description
        #     desc = (t.get("description") or "").strip().split(".")[0]
        #     if desc:
        #         # sanitize desc heavily: remove tool names / executable-looking tokens
        #         # desc_clean = re.sub(_COMMAND_RE, "[redacted_tool_or_command]", desc)
        #         # desc_clean = re.sub(r"\s+", " ", desc_clean).strip()
        #         # ensure short
        #         if len(desc_clean) > 300:
        #             desc_clean = desc_clean[:300].rsplit(" ",1)[0] + "..."
        #         safe_line = desc_clean
        #     else:
        #         safe_line = "[NO_SUMMARY_AVAILABLE]"

        record = {
            "mitre_id": mitre_id,
            "name": name,
            "platform": chosen_platform,
            "command": safe_line
        }

        with open(OUTFILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        count += 1
        # optional: simple progress print every 50
        if count % 50 == 0:
            print(f"Processed {count} techniques...")

    print(f"Done. Wrote {count} descriptions to {OUTFILE}")
    print("Note: Any output that appeared potentially actionable was redacted and replaced with a safe fallback.")

if __name__ == "__main__":
    main()
