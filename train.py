import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler
from datasets import load_dataset
from tqdm.auto import tqdm # Progress bars
import math
import os
import argparse
import logging
import json
import re # For finding checkpoint steps
import glob # For finding checkpoint directories

# Import your model definition (assuming it's in landscape_transformer.py)
from landscape_transformer import LandscapeTransformerLM

# --- Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train Landscape Transformer LM")
    # ... (keep all previous args definitions: dataset, model, training) ...
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name from Hugging Face Hub")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1", help="Dataset config name")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="Tokenizer name or path")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Probability of masking tokens")
    parser.add_argument("--embed_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--ff_dim", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_prototypes", type=int, default=512, help="Number of landscape prototypes")
    parser.add_argument("--prototype_key_dim", type=int, default=64, help="Dimension of prototype keys")
    parser.add_argument("--prototype_value_dim", type=int, default=64, help="Dimension of prototype values")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--output_dir", type=str, default="./landscape_lm_output", help="Output directory for model checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type (linear, cosine, etc.)")
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of warmup steps for scheduler")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log training loss every N steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate on validation set every N steps")
    parser.add_argument("--use_amp", action='store_true', help="Use Automatic Mixed Precision")
    # NEW ARGUMENT for resuming
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from (optional)")


    args = parser.parse_args()
    # Ensure prototype dims are reasonable if not specified (or add specific args)
    if args.prototype_key_dim is None:
        args.prototype_key_dim = args.embed_dim // args.num_heads
    if args.prototype_value_dim is None:
         args.prototype_value_dim = args.embed_dim // args.num_heads
    return args

# --- Helper Functions ---
def setup_logging(output_dir):
    # Check if handlers already exist (e.g., if function is called multiple times)
    logger = logging.getLogger()
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(output_dir, "train.log")),
                logging.StreamHandler()
            ]
        )
    else:
         logging.info("Logger already configured.")


def save_checkpoint(model, optimizer, scheduler, scaler, args, global_step, epoch):
    output_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    os.makedirs(output_path, exist_ok=True)
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
    # Save optimizer and scheduler states
    torch.save(optimizer.state_dict(), os.path.join(output_path, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(output_path, "scheduler.pt"))
    # Save AMP scaler state if used
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(output_path, "scaler.pt"))
    # Save training args
    with open(os.path.join(output_path, "training_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    # Save current training state (epoch, step)
    training_state = {"global_step": global_step, "epoch": epoch}
    with open(os.path.join(output_path, "training_state.json"), 'w') as f:
         json.dump(training_state, f, indent=2)

    logging.info(f"Checkpoint saved to {output_path}")

# --- NEW: Function to find the latest checkpoint ---
def find_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # Extract step numbers and find the max
    steps = [int(re.search(r"checkpoint-(\d+)", c).group(1)) for c in checkpoints if re.search(r"checkpoint-(\d+)", c)]
    if not steps:
         return None
    latest_step = max(steps)
    return os.path.join(output_dir, f"checkpoint-{latest_step}")

# --- Main Training Script ---
def main():
    args = parse_args()

    # --- Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args.output_dir)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Training args: {vars(args)}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0 # Default to 0 if not set
    logging.info(f"Tokenizer loaded: {args.tokenizer_name} (Vocab size: {vocab_size}, Pad ID: {pad_token_id})")

    # --- Dataset Loading & Preprocessing ---
    # ... (Dataset loading and preprocessing remains the same as before) ...
    logging.info("Loading and preprocessing dataset...")
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, trust_remote_code=True) # Added trust_remote_code

    def tokenize_function(examples):
        text_column = "text" if "text" in examples else list(examples.keys())[0]
        # Ensure texts are strings
        texts = [str(t) for t in examples[text_column] if t is not None]
        if not texts:
             return {} # Return empty if no valid text
        return tokenizer(texts, truncation=False) # Don't truncate yet

    # Filter out None types before mapping
    raw_datasets = raw_datasets.filter(lambda example: example.get("text") is not None)

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length < args.max_seq_length:
             return {k: [] for k in examples.keys()} # Skip if not enough tokens
        total_length = (total_length // args.max_seq_length) * args.max_seq_length
        result = {
            k: [t[i : i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["attention_mask"] = [[1] * args.max_seq_length for _ in range(len(result["input_ids"]))] # Assume full length
        return result

    processed_datasets = tokenized_datasets.map(group_texts, batched=True)
    logging.info(f"Dataset processed. Train examples: {len(processed_datasets['train'])}, Val examples: {len(processed_datasets['validation'])}")


    # --- Data Collator for MLM ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability, return_tensors="pt"
    )

    # --- DataLoaders ---
    train_dataloader = DataLoader(
        processed_datasets["train"],
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        pin_memory=True # Add pin_memory for potentially faster GPU transfer
    )
    eval_dataloader = DataLoader(
        processed_datasets["validation"],
        batch_size=args.batch_size,
        collate_fn=data_collator,
        pin_memory=True
    )

    # --- Model Initialization ---
    model = LandscapeTransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        num_prototypes=args.num_prototypes,
        prototype_key_dim=args.prototype_key_dim,
        prototype_value_dim=args.prototype_value_dim,
        dropout=args.dropout,
        max_len=args.max_seq_length,
        pad_token_id=pad_token_id # Pass pad_token_id
    )
    model.to(device)
    logging.info("Model initialized.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of trainable parameters: {num_params / 1e6:.2f} M")

    # --- Optimizer & Scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = args.num_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logging.info("Optimizer and scheduler initialized.")

    # --- Automatic Mixed Precision (AMP) ---
    scaler = None
    if args.use_amp and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        logging.info("Using Automatic Mixed Precision.")

    # --- Checkpoint Loading --- <<< NEW SECTION >>>
    global_step = 0
    start_epoch = 0
    checkpoint_to_resume = args.resume_from_checkpoint
    # If resume path not specified, try finding the latest in output_dir
    if checkpoint_to_resume is None:
         checkpoint_to_resume = find_latest_checkpoint(args.output_dir)

    if checkpoint_to_resume and os.path.isdir(checkpoint_to_resume):
        logging.info(f"Resuming training from checkpoint: {checkpoint_to_resume}")
        model_path = os.path.join(checkpoint_to_resume, "pytorch_model.bin")
        optimizer_path = os.path.join(checkpoint_to_resume, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_to_resume, "scheduler.pt")
        scaler_path = os.path.join(checkpoint_to_resume, "scaler.pt")
        state_path = os.path.join(checkpoint_to_resume, "training_state.json")

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        if scheduler is not None and os.path.exists(scheduler_path):
            scheduler.load_state_dict(torch.load(scheduler_path))
        if scaler is not None and os.path.exists(scaler_path):
            scaler.load_state_dict(torch.load(scaler_path))
        if os.path.exists(state_path):
             with open(state_path, 'r') as f:
                  training_state = json.load(f)
                  global_step = training_state.get("global_step", 0)
                  # Resume from the *next* epoch after the saved one
                  start_epoch = training_state.get("epoch", 0) + 1
                  logging.info(f"  Resumed global_step: {global_step}")
                  logging.info(f"  Resuming from epoch: {start_epoch}")
        # Potentially skip dataloader steps if resuming mid-epoch (more complex, skipping for now)
    else:
         logging.info("No checkpoint found or specified. Starting training from scratch.")


    # --- Training Loop ---
    total_loss = 0.0
    # Make sure progress bar starts correctly if resuming
    completed_steps_in_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    logging.info("***** Starting Training *****")
    # ... (log dataset sizes etc. as before) ...
    logging.info(f"  Num examples = {len(processed_datasets['train'])}")
    logging.info(f"  Num Epochs = {args.num_train_epochs}")
    logging.info(f"  Start Epoch = {start_epoch}")
    logging.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {num_training_steps}")
    logging.info(f"  Starting global step = {global_step}")


    model.train()
    # Update range to start from start_epoch
    for epoch in range(start_epoch, args.num_train_epochs):
        # Handle resuming mid-epoch progress bar
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{args.num_train_epochs}",
            initial=completed_steps_in_epoch * args.gradient_accumulation_steps, # Start progress bar correctly
            total=len(train_dataloader)
        )
        # Skip steps already completed in the resumed epoch
        if epoch == start_epoch and completed_steps_in_epoch > 0:
             logging.info(f"Skipping {completed_steps_in_epoch * args.gradient_accumulation_steps} steps from resumed epoch {epoch+1}")
             # This requires dataloader to be iterable; standard DataLoader is.
             import itertools
             # Consume items from the dataloader iterator
             for _ in itertools.islice(progress_bar, completed_steps_in_epoch * args.gradient_accumulation_steps):
                 pass

        for step, batch in enumerate(progress_bar, start=completed_steps_in_epoch * args.gradient_accumulation_steps):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                # Pass attention_mask from batch to model
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
                logits = outputs

                # Calculate Loss (MLM)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, vocab_size), batch["labels"].view(-1))
                loss = loss / args.gradient_accumulation_steps

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * args.gradient_accumulation_steps

            # Optimizer step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = total_loss / (args.logging_steps * args.gradient_accumulation_steps) # Correct averaging
                    logging.info(f"Step: {global_step}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                    total_loss = 0.0

                # Evaluation
                if global_step % args.eval_steps == 0:
                    evaluate(model, eval_dataloader, device, scaler, args, global_step)
                    model.train()

                # Checkpointing (pass current epoch)
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, scaler, args, global_step, epoch)

            progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})

        # Reset completed steps for the next epoch
        completed_steps_in_epoch = 0


    # --- Final Save ---
    logging.info("Training finished. Saving final model.")
    # Use the final epoch number when saving
    final_epoch = args.num_train_epochs - 1
    save_checkpoint(model, optimizer, scheduler, scaler, args, global_step, final_epoch)

# --- Evaluation Function ---
# (Evaluation function remains the same, but added vocab_size to args for clarity)
def evaluate(model, eval_dataloader, device, scaler, args, global_step):
    logging.info(f"--- Running Evaluation at Step {global_step} ---")
    model.eval()
    total_eval_loss = 0
    eval_steps = 0
    progress_bar = tqdm(eval_dataloader, desc="Evaluating", leave=False) # leave=False for cleaner logs

    with torch.no_grad():
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                # Pass attention_mask from batch to model
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
                logits = outputs

                loss_fct = nn.CrossEntropyLoss()
                # Get vocab size from model config or args if possible
                # Assuming args.vocab_size is correctly passed or retrieved
                vocab_size = model.output_head.out_features # More robust way to get vocab size
                loss = loss_fct(logits.view(-1, vocab_size), batch["labels"].view(-1))

            total_eval_loss += loss.item()
            eval_steps += 1
            progress_bar.set_postfix({"eval_loss": loss.item()})

    avg_eval_loss = total_eval_loss / eval_steps
    try:
        perplexity = math.exp(avg_eval_loss)
    except OverflowError:
        perplexity = float("inf") # Handle potential overflow if loss is too high

    logging.info(f"--- Evaluation Results ---")
    logging.info(f"Step: {global_step}, Average Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}")
    logging.info(f"--------------------------")


# --- Entry Point ---
if __name__ == "__main__":
    main()