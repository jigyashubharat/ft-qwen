import torch
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

# 1. Setup
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = "./qwen_finetuned"

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token # Fix for Qwen missing pad token

# 3. Load & Format Dataset
# We manually format the chat into a single string to avoid SFTTrainer guessing errors
def formatting_prompts_func(example):
    # Apply the chat template (User: ... Assistant: ...)
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return text

dataset = load_dataset("json", data_files="output.jsonl", split="train")

# 4. Load Model (Low Memory Mode)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa" # Use Flash Attention 2 if available, else standard
)

# 5. LoRA Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 6. Training Arguments (Safe Settings)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # Lowest possible to save RAM
    gradient_accumulation_steps=4,     # Accumulate to simulate larger batch
    learning_rate=2e-4,
    logging_steps=5,
    num_train_epochs=1,
    fp16=False,
    report_to="none",
)

# 7. Start Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    formatting_func=formatting_prompts_func, # Explicit formatting
)

print("Starting training...")
trainer.train()
print("Saved to", OUTPUT_DIR)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)