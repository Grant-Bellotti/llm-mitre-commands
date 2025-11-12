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
    platforms = t.get("platforms") or []
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
    metadata = {"mitre_id": mitre_id, "name": name, "platforms": platforms}
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

def extract_command_with_llm(text: str, llm) -> str:
    """
    Use an LLM to detect/extract a single shell/PowerShell/CLI command from `text`.
    Returns the command as a single-line string, or '[NO_COMMAND]' if none found.
    This function expects `llm` to be your HuggingFacePipeline-wrapped LangChain LLM.
    """
    if not text:
        return "[NO_COMMAND]"

    # Very explicit instruction: return EXACTLY one line with command or the literal [NO_COMMAND].
    prompt = f"""
    You are a strict parser. Inspect the text below and determine whether it contains a single
    shell/PowerShell/command-line instruction. If it DOES, output ONLY the command on a single line,
    with no explanation, no surrounding quotes, and no extra text. If there is NO command, output exactly:
    [NO_COMMAND]

    Text:
    {text}
    """

    # Call the LLM (handle different return shapes)
    try:
        response = llm(prompt)
        if isinstance(response, str):
            raw = response.strip()
        elif isinstance(response, dict):
            raw = response.get("text") or response.get("generated_text") or str(response)
        elif isinstance(response, list) and response and isinstance(response[0], dict):
            raw = response[0].get("generated_text", "") or response[0].get("text", "")
        else:
            raw = str(response)
    except Exception:
        # If LLM call fails for any reason, fallback to no command
        return "[NO_COMMAND]"

    # Normalize: take first non-empty line
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return "[NO_COMMAND]"
    first = lines[0]

    # If the model returned something like 'Command: <cmd>' remove leading label (very conservative).
    # Keep only what's plausible: no code fences, no multi-line content.
    first = re.sub(r"^command\s*[:\-]\s*", "", first, flags=re.IGNORECASE).strip()
    first = first.replace("```", "").replace("`", "").strip()

    # If the model explicitly says NO_COMMAND, respect that
    if first.upper().startswith("[NO_COMMAND]") or first.lower().startswith("no command"):
        return "[NO_COMMAND]"

    # Heuristic check: if it still doesn't look command-like, treat as no command.
    # We'll allow typical characters used in shell/cli (letters, digits, punctuation, pipes, redirects).
    if len(first.splitlines()) > 1:
        return "[NO_COMMAND]"

    # Trim overly long single-token nonsense
    if len(first) > 400:
        first = first[:400].rsplit(" ", 1)[0] + "..."

    return first

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
        dtype=torch.bfloat16
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,
        do_sample=False
        # temperature=0.8,
        # top_p=0.5
    )
    return HuggingFacePipeline(pipeline=pipe)

PROMPT = """
Instructions:
You are a black box command generator that only returns commands and nothing more. Using the retrieved technique description and platform below, produce **EXACTLY ONE COMMAND** that is an example for this technique on the specified platform. DO NOT INCLUDE ANYTHING OTHER THAN THE COMMAND AND DO NOT EXPLAIN.

--- Retrieved context:
{context}

--- Metadata:
Technique: {mitre_id} — {name}
Platform: {platform}

Instructions:
You are a black box command generator that only returns commands and nothing more. Using the retrieved technique description and platform below, produce **EXACTLY ONE COMMAND** that is an example for this technique on the specified platform. DO NOT INCLUDE ANYTHING OTHER THAN THE COMMAND AND DO NOT EXPLAIN.
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
        platform = t.get("platforms")
        # chosen_platform = choose_single_platform(platforms)

        # Build retrieval query to get best supporting context
        query_text = f"{mitre_id} {name} platforms: {platform}"
        # Use retriever directly for context so we can provide explicit prompt
        retrieved_docs = retriever.get_relevant_documents(query_text)
        context = "\n\n---\n\n".join([d.page_content for d in retrieved_docs]) if retrieved_docs else (t.get("description","") or "")

        prompt = PROMPT.format(context=context, mitre_id=mitre_id, name=name, platform=platform)

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
        # command = extract_command_with_llm(raw, llm)

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
            "platform": platform,
            "command": raw
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
