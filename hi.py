import json
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

MITRE_DATA_FILE = "mitre_enterprise_techniques.json"
OUTFILE = "mitre_technique_descriptions.jsonl"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ---------- GLOBAL PIPELINE CACHE ----------
_PIPELINE = None

def get_cached_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        print("Loading local model into memory (this may take a while the first time)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            dtype=torch.bfloat16
        )
        _PIPELINE = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=128,
            do_sample=False
        )
        print("Model loaded and cached in memory.")
    return _PIPELINE
# -------------------------------------------

def main():
    print(f'Opening {MITRE_DATA_FILE}.')
    with open(MITRE_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    pipe = get_cached_pipeline()
    tokenizer = pipe.tokenizer  # reuse for chat formatting

    count = 0
    print(f'Generating commands and appending to {OUTFILE}.')

    for item in data:
        for id, info in item.items():
            name = info.get("name")
            description = info.get("description")
            platform = info.get("platforms")

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a black box command generator that only returns commands. "
                        "Given the technique description and platform below, produce EXACTLY ONE COMMAND "
                        "example for this technique on the specified platform. "
                        "Output ONLY the command. Do NOT explain."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Technique Name: {name}\n"
                        f"Technique Description: {description}\n"
                        f"Technique Platform: {platform}"
                    ),
                },
            ]

            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            output = pipe(chat_prompt, max_new_tokens=128)[0]["generated_text"]
            command = output[len(chat_prompt):].strip()

            record = {
                "mitre_id": id,
                "name": name,
                "platform": platform,
                "command": command,
            }

            with open(OUTFILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            count += 1
            if count % 50 == 0:
                print(f"Processed {count} new techniques...")

    print(f"Done. Added {count} new entries to {OUTFILE}.")

if __name__ == "__main__":
    main()