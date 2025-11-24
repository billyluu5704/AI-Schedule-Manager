#!/usr/bin/env python
# coding: utf-8
import os
#os.makedirs(r"D:\hf_cache\hub", exist_ok=True)
os.makedirs(r"~/luudh/MyFile/vr_lab/hf_cache/hub", exist_ok=True)
#os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

# Best: point directly to the hub cache folder
os.environ["HUGGINGFACE_HUB_CACHE"] = r"~/luudh/MyFile/vr_lab/hf_cache/hub"


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset, DatasetDict
from trl import SFTTrainer, setup_chat_format
import pandas as pd
import json
import re


from huggingface_hub import login

login(token = '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')


base_model = "meta-llama/Llama-3.1-8B-Instruct"
#dataset = "~/luudh/MyFile/cohen_lab/tech_company_detailed_tasks.csv"
#fine_tuned_model = "~/luudh/MyFile/cohen_lab/Llama-3.1-8B-Instruct-finetuned"
dataset = os.path.expanduser(
    "~/luudh/MyFile/AI_Scheduling/data/tech_company_detailed_tasks_timeAdjusted_v2_importanceFixed.csv"
)

# put checkpoints in a folder you own (relative to your project)
fine_tuned_model = os.path.expanduser(
    "~/luudh/MyFile/AI_Scheduling/Llama-3.1-8B-Instruct-finetuned-version2"
)
os.makedirs(fine_tuned_model, exist_ok=True)


import wandb

wandb.login(key="7f6a676fe521599b6d6d33827b95ca0432169e0b")
run = wandb.init(
    project="Fine-tune for instruction llama-3.2-3B-Instruct on Tech Company Tasks Dataset Version 2",
    job_type="training",
    anonymous="allow",
    reinit=True
)


#set data type
""" if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager" """
torch_dtype = torch.float16
attn_implementation = "eager"


# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

#load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation,
)



#load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model,trust_remote_code=True)



df = pd.read_csv(dataset)
df.head()


def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # Shuffle the DataFrame
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    #split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df

train_df, validation_df, test_df = random_split(df, 0.85, 0.1)


#convert DataFram to datasets.Dataset
from datasets import Dataset

train_dataset = Dataset.from_pandas(train_df)
validation_dataset = Dataset.from_pandas(validation_df)
test_dataset = Dataset.from_pandas(test_df)


#function for deadline format
from datetime import datetime

def format_deadline(raw_deadline):
    try:
        dt = datetime.strptime(raw_deadline.strip(), "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            dt = datetime.strptime(raw_deadline.strip(), "%m/%d/%Y %H:%M")
        except ValueError:
            dt = datetime.fromisoformat(raw_deadline.strip())
    return dt.strftime("%Y-%m-%d %H:%M")

def calculate_estimated_time(deadline_str):
    # Parse deadline into datetime
    try:
        deadline = datetime.strptime(deadline_str.strip(), "%Y-%m-%d %H:%M")
    except ValueError:
        try:
            deadline = datetime.strptime(deadline_str.strip(), "%m/%d/%Y %H:%M")
        except ValueError:
            deadline = datetime.fromisoformat(deadline_str.strip())

    now = datetime.now()
    delta_days = max((deadline - now).days, 0)

    return delta_days  # return days only


instruction = """You are an assistant that extracts structured metadata from task descriptions.

Return the following fields:
- Task Priority (high, medium, low)
- Task Complexity (1-10 scale)
- Required Skills (list of skills, including both technical and soft/inferred skills)
- Deadline (in YYYY-MM-DD HH:MM format)
- NTS_skills (object with the keys: ambiguity_tolerance, communication, planning, collaboration, reasoning, risk_awareness, ownership, stakeholder_mgmt; each value is an integer 1-5)
- importance (object with the keys: ambiguity_tolerance, communication, planning, collaboration, reasoning, risk_awareness, ownership, stakeholder_mgmt; each value is an integer 1-5)

Respond only with the structured output in JSON format.

Example input:
"Develop an interactive dashboard using React and Chart.js to visualize weekly sales data from multiple regions.
The dashboard should include filter options, responsive design, and export to PDF feature.
You are expected to collaborate with the marketing and BI teams to finalize KPIs.
Initial draft is expected by Wednesday, and final delivery should be completed by 2025-06-25 17:00."

Output:
{
  "Task Priority": "high",
  "Task Complexity": 7,
  "Required Skills": [
    "React",
    "Chart.js",
    "Data Visualization",
    "Responsive Design",
    "PDF Export",
    "Team Collaboration",
    "KPI Analysis"
  ],
  "Deadline": "2025-06-25 17:00",
  "NTS_skills": {
    "ambiguity_tolerance": 3,
    "communication": 4,
    "planning": 4,
    "collaboration": 4,
    "reasoning": 3,
    "risk_awareness": 3,
    "ownership": 4,
    "stakeholder_mgmt": 4
  },
  "importance": {
    "ambiguity_tolerance": 3,
    "communication": 4,
    "planning": 4,
    "collaboration": 4,
    "reasoning": 3,
    "risk_awareness": 3,
    "ownership": 4,
    "stakeholder_mgmt": 4
  }
}
"""

import json

NTS_KEYS = [
    "ambiguity_tolerance",
    "communication",
    "planning",
    "collaboration",
    "reasoning",
    "risk_awareness",
    "ownership",
    "stakeholder_mgmt",
]

def _parse_skill_list(s):
    if isinstance(s, str):
        return [skill.strip() for skill in s.split(",") if skill.strip()]
    return s or []

def _parse_json_col(val, default_keys):
    """
    val: could be a dict already or a JSON string.
    we return a dict with all 8 keys, defaulting to 3 if missing.
    """
    if isinstance(val, dict):
        d = val
    else:
        try:
            d = json.loads(val)
        except Exception:
            d = {}
    out = {}
    for k in default_keys:
        v = d.get(k, 3)
        # clamp to 1..5
        try:
            v = int(v)
        except Exception:
            v = 3
        v = max(1, min(5, v))
        out[k] = v
    return out

def format_chat_template(row):
    # your existing logic
    formatted_deadline = format_deadline(row["Deadline"])
    #delta_days = calculate_estimated_time(formatted_deadline)

    # parse skills
    required_skills = _parse_skill_list(row["Required Skills"])

    # NEW: parse NTS_skills and importance from the row
    nts = _parse_json_col(row.get("NTS_skills", "{}"), NTS_KEYS)
    imp = _parse_json_col(row.get("importance", "{}"), NTS_KEYS)

    label = {
        "Task Priority": row["Task Priority"],
        "Task Complexity": row["Task Complexity"],
        "Required Skills": required_skills,
        "Deadline": formatted_deadline,
        "NTS_skills": {k: nts[k] for k in NTS_KEYS},
        "importance": {k: imp[k] for k in NTS_KEYS},
    }

    row_json = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": row["input"]},
        {"role": "assistant", "content": json.dumps(label, indent=2)},
    ]

    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row


train_dataset = train_dataset.map(
    format_chat_template,
    num_proc=4,
)

train_dataset["text"][3]



validation_dataset = validation_dataset.map(
    format_chat_template,
    num_proc=4,
)
test_dataset = test_dataset.map(
    format_chat_template,
    num_proc=4,
)



import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_modules_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_modules_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_modules_names: #needed for 16bit
        lora_modules_names.remove('lm_head')
    return list(lora_modules_names)

modules = find_all_linear_names(model)



# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules
)

tokenizer.chat_template = None  # Reset chat template to avoid issues with the tokenizer
model, tokenizer = setup_chat_format(model, tokenizer)
model = get_peft_model(model, peft_config)



#Hyperparameters for training
training_args = TrainingArguments(
    output_dir=fine_tuned_model,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=5,  # Start with 5, monitor
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=10,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-5,
    fp16=False,
    bf16=False,
    group_by_length=False,
    report_to="wandb",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
)



from transformers.trainer_utils import get_last_checkpoint

#setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    peft_config=peft_config,
    args=training_args,
)



if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
else:
    last_checkpoint = None

#train the model
if last_checkpoint is not None:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Starting training from scratch.")
    trainer.train()

#save the model
trainer.save_model(fine_tuned_model)
tokenizer.save_pretrained(fine_tuned_model)


wandb.finish()
model.config.use_cache = True




logs = trainer.state.log_history



import matplotlib.pyplot as plt

loss_steps = []
loss_values = []
eval_steps = []
eval_losses = []
eval_accuracies = []

for entry in logs:
    if "loss" in entry and "step" in entry:
        loss_steps.append(entry["step"])
        loss_values.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_losses.append(entry["eval_loss"])
        eval_accuracies.append(entry.get("eval_accuracy", None))

# Plot training and eval loss
plt.figure(figsize=(10, 5))
plt.plot(loss_steps, loss_values, label="Training Loss")
plt.plot(eval_steps, eval_losses, label="Eval Loss", marker="o")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss over Training Steps")
plt.grid()
plt.show()

#Plot eval accuracy
if any(eval_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(eval_steps, eval_accuracies, label="Eval Accuracy", marker="o", color="green")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy over Steps")
    plt.grid()
    plt.legend()
    #plt.show()


plt.savefig("/home/luudh/luudh/MyFile/AI_Scheduling/plots/instruction_loss_curve.png", dpi=300)




