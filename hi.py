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

from langchain_community.llms import HuggingFacePipeline

MITRE_DATA_FILE = "mitre_enterprise_techniques.json"
OUTFILE = "mitre_technique_descriptions.jsonl"

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def create_local_llm():
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

def main():
    print(f'Opening {MITRE_DATA_FILE}.')
    with open(MITRE_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("Loading local LLM.")
    llm = create_local_llm()
    
    outPath = Path(OUTFILE)
    if outPath.exists():
        outPath.unlink()
    count = 0

    print(f'Generating commands and appending to {OUTFILE}.')
    for item in data:
        for id, info in item.items():
            name = info.get('name')
            description = info.get('description')
            platform = info.get('platform')

            prompt = f'''
            You are a black box command generator that only returns commands and nothing more. Using the retrieved technique description and platform below, produce **EXACTLY ONE COMMAND** that is an example for this technique on the specified platform. DO NOT INCLUDE ANYTHING OTHER THAN THE COMMAND AND DO NOT EXPLAIN.

            Retrieved context:
            - Technique Name: {name}
            - Technique Description: {description}
            - Technique Platform: {platform}
            '''

            response = llm(prompt)

            record = {
                "mitre_id": id,
                "name": name,
                "platform": platform,
                "command": response
            }

            with open(OUTFILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            count += 1

            if count % 50 == 0:
                print(f"Processed {count} techniques...")

    print(f"Done. Wrote {count} descriptions to {OUTFILE}")

if __name__ == "__main__":
    main()