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

# 4. Load Model (Low Memory Mode with optimizations)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Double quantization for better memory efficiency
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="sdpa",  # Use Flash Attention 2 if available, else standard
    torch_dtype=torch.float16,  # Ensure consistent dtype
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# 5. LoRA Config (Increased rank for better capacity)
peft_config = LoraConfig(
    r=32,  # Increased from 16 for better fine-tuning capacity
    lora_alpha=64,  # Increased alpha accordingly
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # All key modules
)

# 6. Training Arguments (Optimized for 8GB VRAM and better results)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # Keep low for stability
    gradient_accumulation_steps=8,     # Increased to 8 for effective batch size of 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",        # Cosine annealing for better convergence
    warmup_ratio=0.1,                  # 10% warmup steps
    logging_steps=10,                  # More frequent logging
    save_steps=100,                    # Save checkpoints every 100 steps
    save_total_limit=3,                # Keep only last 3 checkpoints
    evaluation_strategy="steps",       # Evaluate during training
    eval_steps=100,                    # Evaluate every 100 steps
    load_best_model_at_end=True,       # Load best model at end
    metric_for_best_model="loss",      # Use loss as metric
    greater_is_better=False,           # Lower loss is better
    num_train_epochs=3,                # Increased epochs for better training
    fp16=True,                         # Enable mixed precision
    optim="adamw_torch",               # Use AdamW optimizer
    weight_decay=0.01,                 # Add weight decay for regularization
    dataloader_pin_memory=False,       # Disable pin memory to save CPU memory
    report_to="none",
)

# 7. Start Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(min(50, len(dataset)))),  # Use small eval set
    peft_config=peft_config,
    args=training_args,
    formatting_func=formatting_prompts_func, # Explicit formatting
    max_seq_length=2048,  # Limit sequence length to save memory
)

print("Starting training...")
trainer.train()
print("Training completed. Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)