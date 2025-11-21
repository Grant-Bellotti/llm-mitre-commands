import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

MITRE_DATA_FILE = "mitre_enterprise_techniques.json"
OUTFILE = "mitre_technique_descriptions.jsonl"
# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ---------- Pipeline Cache ----------
_PIPELINE = None

def getCachedPipeline():
    global _PIPELINE

    if _PIPELINE is None:
        print("Loading local model into memory...")

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
            max_new_tokens=256,
            do_sample=False
        )

        print("Model loaded and cached in memory.")

    return _PIPELINE
# -------------------------------------------

# Call the llm with a prompt and return the output
def generateText(prompt, pipe, tokenizer):
    chatPrompt = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )

    output = pipe(chatPrompt, max_new_tokens=128)[0]["generated_text"]
    return (output[len(chatPrompt):].strip())

def main():
    print(f'Opening {MITRE_DATA_FILE}.')
    # Open the Mitre Techniques file
    with open(MITRE_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    pipe = getCachedPipeline()
    tokenizer = pipe.tokenizer

    count = 0
    print(f'Generating commands and appending to {OUTFILE}.')

    # Create a prompt for each Mitre attack technique and have the LLM generate a command
    for item in data:
        for id, info in item.items():
            name = info.get("name")
            description = info.get("description")
            platform = info.get("platforms")

            # Prompt to generate the command
            commandPrompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a black box command generator that only returns commands. "
                        "Given the technique name, description, and platform below, produce EXACTLY ONE COMMAND "
                        "example for this technique on the specified platform. "
                        "Use your knowledge of the platform and it's commands to create a new command that you have not seen before."
                        "If giving an example using a Python script, write the python code as a one liner command."
                        "Output ONLY the command. Do NOT explain or include code blocks."
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

            command = generateText(commandPrompt, pipe, tokenizer)

            # Prompt to verify and fix the command if needed
            checkCommandPrompt = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional command verifier whose job is to check and fix incorrect commands. "
                        "Given the command below, correct the command if needed. "
                        "If the command is already syntatically correct, output the exact same command. "
                        "Output ONLY the ONE command. Do NOT explain or include code blocks."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Command: {command}\n"
                    ),
                },
            ]

            verifiedCommand = generateText(checkCommandPrompt, pipe, tokenizer)

            # Sometimes the 2nd LLM for checking the command correctness says it is unable to create the command because it may cause harm. Switch to first one if so.
            if "can't" in verifiedCommand or "cannot" in verifiedCommand:
                verifiedCommand = command

            # remove ` if they're there
            if verifiedCommand.startswith("`") and verifiedCommand.endswith("`"):
                verifiedCommand = verifiedCommand[1:-1]

            # Create the record to write to data file
            record = {
                "ID": id,
                "Name": name,
                "Platform": platform,
                "Command": verifiedCommand,
            }

            with open(OUTFILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            count += 1
            if count % 50 == 0:
                print(f"Processed {count} new techniques...")

    print(f"Done. Added {count} new entries to {OUTFILE}.")

if __name__ == "__main__":
    main()