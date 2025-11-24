import os, gc
os.makedirs(r"~/luudh/MyFile/vr_lab/hf_cache/hub", exist_ok=True)

# Best: point directly to the hub cache folder
os.environ["HUGGINGFACE_HUB_CACHE"] = r"~/luudh/MyFile/vr_lab/hf_cache/hub"

from huggingface_hub import login

login(token = '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')
base_model = "meta-llama/Llama-3.1-8B-Instruct"
fine_tuned_model = "/home/luudh/luudh/MyFile/AI_Scheduling/Llama-3.1-8B-Instruct-finetuned-version2"

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch
import re
from trl import setup_chat_format
import json

def start_model(base, fine_tuned):
    #reload tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    #print(tokenizer.chat_template)

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    max_memory = {0: "46GiB", "cpu": "64GiB"} # adjust according to your GPU

    """
    max_memory = {
        0: "6GiB",         
        "cpu": "32GiB"
    }
    """
    base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",                  # let it split GPU/CPU
        attn_implementation="eager",        # avoid flash-attn
        torch_dtype=torch.float16,
        max_memory=max_memory,
    )

    tokenizer.chat_template = None
    base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)

    #merge adapter with base model
    model = PeftModel.from_pretrained(base_model_reload, fine_tuned_model)

    model = model.merge_and_unload()
    return model, tokenizer

def extract_json(text: str):
    """
    Scan the text and return the first substring that is valid JSON.
    Handles nested braces and ignores non-JSON code blocks.
    """
    n = len(text)
    for start in range(n):
        if text[start] != "{":
            continue

        depth = 0
        for end in range(start, n):
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start:end+1].strip()

                    # Remove code fences if somehow included
                    if snippet.startswith("```"):
                        snippet = re.sub(r"^```[a-zA-Z]*\n?", "", snippet)
                        snippet = re.sub(r"```$", "", snippet).strip()

                    # Fix Python-style literals if model used them
                    snippet = (
                        snippet.replace("True", "true")
                               .replace("False", "false")
                               .replace("None", "null")
                    )

                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        # Not valid JSON, break inner loop and try next '{'
                        break

    return None

    
def generate(user_input, base_model=base_model, fine_tuned_model=fine_tuned_model):
    model, tokenizer = start_model(base_model, fine_tuned_model)

    instruction = """
        You are an assistant that extracts structured metadata from task descriptions.
        Return the following fields:
        - Task Priority (high, medium, low)
        - Task Complexity (1-10 scale)
        - Required Skills (list of skills, including both technical and soft/inferred skills)
        - Deadline (in YYYY-MM-DD HH:MM format)
        - NTS_skills (object with the keys: ambiguity_tolerance, communication, planning, collaboration, reasoning, risk_awareness, ownership, stakeholder_mgmt; each value is an integer 1-5)
        - importance (object with the keys: ambiguity_tolerance, communication, planning, collaboration, reasoning, risk_awareness, ownership, stakeholder_mgmt; each value is an integer 1-5)

        Respond only with a SINGLE JSON object and NOTHING else:
        no explanations, no markdown, no code.
    """

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    #text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    data = extract_json(text)
    if data is None:
        # If you really only want JSON printed, you can raise instead
        raise ValueError(f"Model did not return valid JSON.\nRaw text:\n{text}")


    # ✅ Unwrap "task": { ... } if present
    if isinstance(data, dict) and "task" in data and isinstance(data["task"], dict):
        data = data["task"]

    data = json.dumps(data, indent=2)
    print(data)
    return data

user_input = """Design a basic churn-prediction pipeline with scikit-learn or XGBoost.
Provide ROC-AUC and F1 after each run and wire a simple nightly Airflow DAG.
This is a medium-priority task with moderate scope (expected effort ~5 days).
The project must be delivered by 2026-01-18 17:00."""

""" if __name__ == "__main__":
    result = generate(user_input)  """
