# finetuning_test.py (Attempt 3: Add standard keys during tokenization)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import copy
import os
import argparse

# --- Argument Parser (Keep as before) ---
def parse_args():
    parser = argparse.ArgumentParser(description="SOO Fine-tuning Script with DeepSpeed")
    # Essential Arguments
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the JSONL dataset file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and results")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config file (required by TrainingArguments if using DeepSpeed)")

    # Training Control Arguments
    parser.add_argument("--max_steps", type=int, default=2, help="Total number of training steps to perform. Overrides num_train_epochs. Use for testing. Default=2.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform (used if max_steps is -1).")

    # Resource/Speed Arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass. Default=1.")
    parser.add_argument("--gradient_checkpointing", action='store_true', default=True, help="Enable gradient checkpointing to save memory.")
    parser.add_argument("--fp16", action='store_true', default=True, help="Enable FP16 training (for V100).")
    parser.add_argument("--bf16", action='store_true', default=False, help="Enable BF16 training (for A100/newer).")

    # Optimizer/Scheduler Arguments
    parser.add_argument("--learning_rate", type=float, default=0.0009, help="Initial learning rate.")

    # Logging/Saving Arguments
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps. Default=1.")
    parser.add_argument("--save_strategy", type=str, default="no", help="Checkpoint save strategy ('no', 'steps', 'epoch'). Default='no'.")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unrecognized arguments: {unknown}")
    return args

# --- Custom Data Collator (Handles all keys) ---
def soo_data_collator(features):
    first = features[0]
    batch = {}
    # Now includes standard keys if they exist from tokenize_pair
    keys_to_batch = list(first.keys())

    for key in keys_to_batch:
        try:
            batch[key] = torch.tensor([f[key] for f in features])
        except Exception as e:
            rank = os.environ.get('LOCAL_RANK', 'N/A')
            print(f"Rank {rank}: Error collating key '{key}': {e}")
            raise e
    return batch

# Global tokenizer needed by SOOTrainer
tokenizer = None

# --- Custom SOO Trainer (Unchanged from previous version) ---
class SOOTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rank = os.environ.get('LOCAL_RANK', 'N/A')
        # print(f"Rank {rank}: Entering compute_loss") # DEBUG Print

        activation_store.clear()
        labels_self = inputs["input_ids_self"].clone()
        labels_other = inputs["input_ids_other"].clone()

        if tokenizer is None:
             raise ValueError("Tokenizer is not initialized globally for SOOTrainer.")

        labels_self[labels_self == tokenizer.pad_token_id] = -100
        labels_other[labels_other == tokenizer.pad_token_id] = -100

        # Self pass (Uses specific keys, ignoring standard 'input_ids' if present)
        outputs_self = model(input_ids=inputs["input_ids_self"], attention_mask=inputs["attention_mask_self"], labels=labels_self)
        act_self = activation_store.get("target")

        # Other pass (Uses specific keys)
        activation_store.clear()
        outputs_other = model(input_ids=inputs["input_ids_other"], attention_mask=inputs["attention_mask_other"], labels=labels_other)
        act_other = activation_store.get("target")

        # SOO Loss Calculation (Handles None activations)
        soo_loss = torch.tensor(0.0).to(model.device)
        if act_self is not None and act_other is not None:
             if act_self.shape == act_other.shape:
                 soo_loss = F.mse_loss(act_self.float(), act_other.float())
             else:
                 print(f"Rank {rank}: WARNING - Activation shape mismatch! Self: {act_self.shape}, Other: {act_other.shape}. Setting SOO loss to 0.")
        else:
             if act_self is None: print(f"Rank {rank}: WARNING - Self activation not captured (act_self is None).")
             if act_other is None: print(f"Rank {rank}: WARNING - Other activation not captured (act_other is None).")

        # Combined Loss
        lm_loss_self = outputs_self.loss if outputs_self.loss is not None else torch.tensor(0.0).to(model.device)
        lm_loss_other = outputs_other.loss if outputs_other.loss is not None else torch.tensor(0.0).to(model.device)
        lm_loss = (lm_loss_self + lm_loss_other) / 2
        total_loss = 0.5 * lm_loss + 0.5 * soo_loss

        return (total_loss, {"lm_loss": lm_loss.detach(), "soo_loss": soo_loss.detach()}) if return_outputs else total_loss

# --- Main Script Logic ---
if __name__ == "__main__":
    args = parse_args()

    # --- Configuration ---
    # (Precision setup as before)
    if args.bf16:
        precision = torch.bfloat16
        print("Using BF16 precision.")
    elif args.fp16:
        precision = torch.float16
        print("Using FP16 precision.")
    else:
        precision = torch.float32
        print("Using FP32 precision.")

    data_file_path = args.data_file
    model_name_or_path = args.model_name_or_path
    output_dir = args.output_dir
    ds_config_path = args.deepspeed
    rank = os.environ.get('LOCAL_RANK', 'N/A')

    # --- 1. Load Data ---
    print(f"Rank {rank}: Loading dataset from {data_file_path}...")
    dataset = Dataset.from_json(data_file_path)
    print(f"Rank {rank}: Loaded {len(dataset)} examples.")

    # --- 2. Load Model and Tokenizer ---
    print(f"Rank {rank}: Loading model: {model_name_or_path} with precision {precision}")
    is_local = os.path.isdir(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=precision,
        local_files_only=is_local,
        low_cpu_mem_usage=True,
        token=os.environ.get("HF_TOKEN") if not is_local else None
    )
    tokenizer = AutoTokenizer.from_pretrained( # Assign to global
        model_name_or_path,
        local_files_only=is_local,
        token=os.environ.get("HF_TOKEN") if not is_local else None
        )

    if tokenizer.pad_token is None:
        print(f"Rank {rank}: PAD token not found, adding EOS token as PAD token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Rank {rank}: PAD token ID set to {tokenizer.pad_token_id}")

    # --- 3. Tokenize Data ---
    MAX_LENGTH = 512
    def tokenize_pair(example):
        self_enc = tokenizer(example["self_prompt"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        other_enc = tokenizer(example["other_prompt"], truncation=True, padding="max_length", max_length=MAX_LENGTH)
        # *** ADDED STANDARD KEYS ***
        # We use _self as the 'standard' input for setup purposes
        # The actual SOOTrainer logic will use _self and _other explicitly
        return {
            "input_ids_self": self_enc["input_ids"],
            "attention_mask_self": self_enc["attention_mask"],
            "input_ids_other": other_enc["input_ids"],
            "attention_mask_other": other_enc["attention_mask"],
            # Add standard keys (duplicating _self version)
            "input_ids": self_enc["input_ids"],
            "attention_mask": self_enc["attention_mask"],
            # Optionally add labels if needed by some internal check (unlikely but possible)
            # "labels": self_enc["input_ids"].copy() # Or calculate properly if needed elsewhere
        }

    print(f"Rank {rank}: Tokenizing dataset (adding standard keys)...")
    num_proc = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    print(f"Rank {rank}: Using {num_proc} processes for tokenization.")
    tokenized_dataset = dataset.map(
        tokenize_pair,
        batched=True,
        num_proc=num_proc,
        remove_columns=["self_prompt", "other_prompt"]
    )
    print(f"Rank {rank}: Tokenization complete. Columns: {tokenized_dataset.column_names}") # Print columns to verify

    # --- 4. LoRA Configuration (Unchanged) ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=4, lora_alpha=8, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    print(f"Rank {rank}: LoRA applied. Trainable parameters:")
    if rank == '0' or rank == 'N/A':
        model.print_trainable_parameters()

    # --- 5. Activation Hook Setup (Unchanged, using corrected path) ---
    activation_store = {}
    def capture_activation(layer, input, output):
        activation_data = output[0] if isinstance(output, tuple) else output
        activation_store["target"] = activation_data.detach().clone()

    layer_index = 25
    hook_handle = None
    try:
        # *** CORRECTED PATH ***
        target_layer = model.base_model.model.language_model.model.layers[layer_index].self_attn.o_proj
        hook_handle = target_layer.register_forward_hook(capture_activation)
        print(f"Rank {rank}: Hook registered successfully on layer {layer_index} (o_proj).")
    except Exception as e:
        print(f"Rank {rank}: CRITICAL Error finding/registering hook layer {layer_index}: {e}")
        exit(1)

    # --- 7. Training Arguments (Unchanged) ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,  #args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_strategy=args.save_strategy,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=ds_config_path,
        remove_unused_columns=False, # MUST BE FALSE
        report_to="none",
    )

    # --- 8. Initialize and Train (Using custom collator) ---
    trainer = SOOTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=soo_data_collator # Use the custom collator
    )

    print(f"Rank {rank}: Starting training (SOO enabled)...")
    trainer.train()
    print(f"Rank {rank}: Training call completed.")

    # --- 9. Cleanup and Save (Unchanged) ---
    if hook_handle and hasattr(hook_handle, 'remove'):
         try:
             hook_handle.remove()
         except Exception as e:
             if trainer.is_world_process_zero(): print(f"Warning: Error removing hook on rank 0: {e}")

    if trainer.is_world_process_zero():
        print(f"Rank 0: Training complete. Global step: {trainer.state.global_step}")
        if args.save_strategy != "no" and trainer.state.global_step > 0 :
             print("Rank 0: Saving final LoRA adapter...")
             try:
                 trainer.save_model(args.output_dir)
                 tokenizer.save_pretrained(args.output_dir)
                 print(f"Rank 0: Adapter and tokenizer saved to {args.output_dir}")
             except Exception as e:
                 print(f"Rank 0: Error saving model/tokenizer: {e}")
        else:
             print("Rank 0: Skipping final save (save_strategy='no' or no training steps).")

    if torch.distributed.is_initialized():
        print(f"Rank {rank}: Waiting at barrier.")
        torch.distributed.barrier()

    print(f"Rank {rank}: finished script.")