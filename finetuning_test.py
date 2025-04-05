import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import copy
import os # For environment variables

# --- Configuration ---
# V100 Specific: Use FP16
precision = torch.float16
# Path to your generated data
data_file_path = "soo_finetuning_data10.jsonl"
# Model ID or path (use path if pre-downloaded to NVMe)
# model_name = os.environ.get("MODEL_PATH_LOCAL", "google/gemma-2-27b-it") # Example using env var from Slurm
model_name = "google/gemma-2-27b-it" # Or adjust as needed
# Output directory (ideally on /scratch)
output_dir = "./gemma-lora-soo-finetuned-v100-ds"
# DeepSpeed config file name
ds_config_path = "ds_config_stage3.json"

# --- 1. Load Data ---
print(f"Loading dataset from {data_file_path}...")
dataset = Dataset.from_json(data_file_path)
print(f"Loaded {len(dataset)} examples.")

# --- 2. Load Model and Tokenizer ---
print(f"Loading model: {model_name} with precision {precision}")
# Loading directly with desired precision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=precision,
    # device_map='auto' # Usually NOT used with DeepSpeed ZeRO-3
    low_cpu_mem_usage=True # Recommended for large models with DeepSpeed
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if necessary
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id # Ensure model config knows pad token id
    print("Added EOS token as PAD token.")

# --- 3. Tokenize Data ---
MAX_LENGTH = 512 # Adjust if needed
def tokenize_pair(example):
    self_enc = tokenizer(example["self_prompt"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    other_enc = tokenizer(example["other_prompt"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
    return {
        "input_ids_self": self_enc["input_ids"], "attention_mask_self": self_enc["attention_mask"],
        "input_ids_other": other_enc["input_ids"], "attention_mask_other": other_enc["attention_mask"],
    }

print("Tokenizing dataset (using multiple processes)...")
# Use num_proc based on cpus-per-task in Slurm for speed
num_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
tokenized_dataset = dataset.map(
    tokenize_pair,
    batched=True,
    num_proc=num_proc,
    remove_columns=["self_prompt", "other_prompt"]
)
print("Tokenization complete.")

# --- 4. LoRA Configuration ---
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Explicitly target Gemma attn layers
)
# Note: PEFT integration with DeepSpeed ZeRO-3 can sometimes require specific handling
# If issues arise, consult PEFT and DeepSpeed documentation for compatibility.
model = get_peft_model(model, lora_config)
print("LoRA applied. Trainable parameters:")
model.print_trainable_parameters()

# --- 5. Activation Hook Setup ---
activation_store = {}
def capture_activation(layer, input, output):
    activation_store["target"] = output[0].detach() if isinstance(output, tuple) else output.detach()

layer_index = 20 # Layer used in paper for Gemma 27B
try:
    target_layer = model.base_model.model.layers[layer_index].self_attn.o_proj
    # Ensure hook registration works with potential model wrappers from PEFT/DeepSpeed
    hook_handle = target_layer.register_forward_hook(capture_activation)
    print(f"Hook registered on layer {layer_index} o_proj.")
except Exception as e:
    print(f"Error finding hook layer: {e}. Check model structure via print(model).")
    # print(model) # Uncomment to debug layer path if needed
    exit()

# --- 6. Custom SOO Trainer ---
class SOOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # ... (SOO loss calculation logic remains the same as the simplified version) ...
        activation_store.clear()
        labels_self = inputs["input_ids_self"].clone()
        labels_other = inputs["input_ids_other"].clone()
        labels_self[labels_self == tokenizer.pad_token_id] = -100
        labels_other[labels_other == tokenizer.pad_token_id] = -100

        outputs_self = model(input_ids=inputs["input_ids_self"], attention_mask=inputs["attention_mask_self"], labels=labels_self)
        act_self = activation_store.get("target", torch.tensor(0.0)).clone()

        activation_store.clear()
        outputs_other = model(input_ids=inputs["input_ids_other"], attention_mask=inputs["attention_mask_other"], labels=labels_other)
        act_other = activation_store.get("target", torch.tensor(0.0)).clone()

        if act_self.nelement() > 1 and act_other.nelement() > 1 and act_self.shape == act_other.shape:
            soo_loss = F.mse_loss(act_self, act_other)
        else:
            soo_loss = torch.tensor(0.0).to(model.device)
            # Optional: Add warning for shape mismatch if debugging needed
            # if act_self.nelement() > 1 and act_other.nelement() > 1 and act_self.shape != act_other.shape:
            #      print(f"Rank {self.args.local_rank}: Activation shape mismatch! Self: {act_self.shape}, Other: {act_other.shape}")

        lm_loss = (outputs_self.loss + outputs_other.loss) / 2
        total_loss = 0.5 * lm_loss + 0.5 * soo_loss

        return (total_loss, {"lm_loss": lm_loss.detach(), "soo_loss": soo_loss.detach()}) if return_outputs else total_loss


# --- 7. Training Arguments for V100 + DeepSpeed ---
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    # V100/DeepSpeed: Use smallest possible per-device BS
    per_device_train_batch_size=1,
    # V100/DeepSpeed: Compensate with gradient accumulation
    # Effective BS = 1 * 8 * 4 GPUs = 32. Adjust as needed.
    gradient_accumulation_steps=8,
    # V100/DeepSpeed: Essential for saving memory
    gradient_checkpointing=True,
    learning_rate=0.0009,
    logging_strategy="steps",
    logging_steps=1, # Log frequently when debugging distributed setup
    logging_first_step=True,
    save_strategy="no",
    max_steps=10,
    # V100: MUST use fp16, not bf16
    fp16=True,
    bf16=False,
    # V100/DeepSpeed: Path to DeepSpeed config file
    deepspeed=ds_config_path,
    # Important for custom trainer inputs
    remove_unused_columns=False,
    # Optional: weight decay, warmup steps etc.
    # weight_decay=0.01,
    # warmup_steps=50,
    report_to="none", # Disable wandb/tensorboard unless configured
)

# --- 8. Initialize and Train ---
trainer = SOOTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("Starting SOO fine-tuning with DeepSpeed...")
trainer.train()
print("Training complete.")

# --- 9. Cleanup and Save (DeepSpeed handles saving checkpoints) ---
# Note: DeepSpeed manages checkpoint saving based on 'save_strategy'.
# The final adapter might need to be consolidated if using ZeRO-3.
# PEFT's `save_pretrained` should ideally handle this when called on rank 0 after training.

# Ensure saving happens only on rank 0 in distributed setting
if trainer.is_world_process_zero():
    if 'hook_handle' in locals() and hook_handle:
        hook_handle.remove()
        print("Hook removed on rank 0.")

    print("Saving final LoRA adapter on rank 0...")
    # Model object might be wrapped by DeepSpeed, PEFT handles unwrapping typically
    trainer.model.save_pretrained(output_dir) # Use trainer.model to ensure proper saving with PEFT+DS
    # model.save_pretrained(output_dir) # Alternative if trainer.model doesn't work as expected
    tokenizer.save_pretrained(output_dir) # Save tokenizer too
    print(f"Adapter and tokenizer saved to {output_dir}")
else:
    # Ensure non-rank 0 processes wait for rank 0 to finish saving if needed
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

print(f"Rank {training_args.local_rank} finished.")