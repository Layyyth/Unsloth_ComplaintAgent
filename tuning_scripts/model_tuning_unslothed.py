#!/usr/bin/env python3
"""
Pod-optimized training script for banking complaint classification and data extraction
using Unsloth for memory-efficient fine-tuning.

Author: Laith
Date: 2023-10-27
"""

import os
import json
import torch
import pandas as pd
import sys
import logging
import random
import numpy as np
import argparse
import re
from typing import Dict, List, Tuple
from datetime import datetime
import warnings

# Suppress warnings for a cleaner log
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. SETUP LOGGING AND REPRODUCIBILITY
# ==============================================================================
def setup_logging():
    """Configures logging to both file and console."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("training_run.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Silence overly verbose loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

logger = setup_logging()

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed} for reproducibility.")

set_seed(42)

# Now, we can safely import the heavy libraries
try:
    from datasets import Dataset, DatasetDict
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from sklearn.metrics import (
        classification_report,
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    import wandb
    from trl import SFTTrainer
    from transformers import TrainingArguments
    logger.info("✓ All ML libraries imported successfully.")
except ImportError as e:
    logger.error(f"❌ Failed to import a required ML library: {e}")
    logger.error("Please install all dependencies from requirements.txt")
    sys.exit(1)


# ==============================================================================
# 2. SYSTEM PROMPT AND CONFIGURATION
# ==============================================================================
SYSTEM_PROMPT = """You are an automated banking customer service ticket analysis system. Your purpose is to parse a customer's request and structure it into a standardized JSON format for internal ticketing.

You must perform the following actions:
1.  Carefully analyze the user's input to understand their intent and key details.
2.  Populate all fields in the JSON object based *only* on the user's text. Do not invent information.
3.  Adhere strictly to the defined categories for `ticket_type`, `severity`, and other categorical fields.
4.  If the user's request is NOT related to banking or financial services (e.g., tech support for a personal computer, dating advice), you MUST reject it by responding with `{"error": "Request is outside the banking support domain."}`.

Your entire response must be ONLY the JSON object, with no conversational text, apologies, or explanations.

The required JSON format is:
{
  "ticket_type": "complaint" | "inquiry" | "assistance",
  "title": "A brief, descriptive summary of the user's issue.",
  "description": "A more detailed description based on the user's full input.",
  "severity": "low" | "medium" | "high" | "critical",
  "department_impacted": "The most relevant bank department.",
  "service_impacted": "The specific banking service affected."
}"""

class ComplaintClassifierTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_wandb()

    def setup_wandb(self):
        """Initializes Weights & Biases if the API key is available."""
        if os.getenv("WANDB_API_KEY"):
            try:
                wandb.init(
                    project=self.config.get("wandb_project", "banking-complaint-classifier"),
                    name=f"{self.config['model_name'].split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M')}",
                    config=self.config,
                    resume="allow"
                )
                logger.info("✓ W&B initialized successfully.")
            except Exception as e:
                logger.warning(f"⚠️ W&B initialization failed: {e}. Training will continue without W&B.")
                self.config['report_to'] = "none"
        else:
            logger.info("WANDB_API_KEY not found. Skipping W&B initialization.")
            self.config['report_to'] = "none"

    def check_gpu_compatibility(self):
        """Checks for CUDA and GPU compatibility."""
        if not torch.cuda.is_available():
            logger.error("❌ CUDA is not available. This script requires a GPU-enabled pod.")
            raise SystemExit("Exiting: No GPU found.")
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✓ GPU Found: {gpu_name}")
        logger.info(f"✓ GPU Memory: {gpu_memory_gb:.2f} GB")
        logger.info(f"✓ CUDA Version: {torch.version.cuda}")
        logger.info(f"✓ bfloat16 Supported: {is_bfloat16_supported()}")
        if gpu_memory_gb < 15:
            logger.warning("⚠️ GPU memory is less than 15GB. Consider reducing batch size if you encounter memory errors.")

    def load_and_prepare_data(self, data_path: str) -> Tuple[List, List, List]:
        """Loads, analyzes, cleans, and splits the data."""
        logger.info("=== Data Loading and Preparation Pipeline ===")
        # Load
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✓ Loaded {len(data)} samples from {data_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load or parse data from {data_path}: {e}")
            raise

        # Analyze and Clean
        unique_data = []
        seen_inputs = set()
        for sample in data:
            if 'user_input' in sample and sample['user_input'] and sample['user_input'] not in seen_inputs:
                unique_data.append(sample)
                seen_inputs.add(sample['user_input'])
        
        duplicate_count = len(data) - len(unique_data)
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate samples ({duplicate_count/len(data):.1%}).")
        
        # Stratified Split
        type_groups = {}
        for sample in unique_data:
            ticket_type = sample.get('ticket_data', {}).get('ticket_type', 'unknown')
            type_groups.setdefault(ticket_type, []).append(sample)
        
        train_samples, val_samples, test_samples = [], [], []
        train_ratio, val_ratio = 0.8, 0.1 # Test ratio is implicitly 0.1
        
        logger.info("Performing stratified split based on 'ticket_type':")
        for ticket_type, samples in type_groups.items():
            random.shuffle(samples)
            n_samples = len(samples)
            train_end = int(n_samples * train_ratio)
            val_end = train_end + int(n_samples * val_ratio)
            
            train_samples.extend(samples[:train_end])
            val_samples.extend(samples[train_end:val_end])
            test_samples.extend(samples[val_end:])
            logger.info(f"  - {ticket_type}: {len(samples[:train_end])} train, {len(samples[train_end:val_end])} val, {len(samples[val_end:])} test")

        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        logger.info(f"✓ Final split sizes -> Train: {len(train_samples)}, Validation: {len(val_samples)}, Test: {len(test_samples)}")
        return train_samples, val_samples, test_samples

    def get_model_and_tokenizer(self):
        """Loads the 4-bit quantized model and tokenizer using Unsloth."""
        logger.info(f"🚀 Initializing model: {self.config['model_name']}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config["model_name"],
            max_seq_length=self.config["max_seq_length"],
            dtype=None,      # Let Unsloth auto-detect dtype
            load_in_4bit=True,
        )

        logger.info("Applying PEFT (LoRA) configuration...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config["lora_r"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
        )
        
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="qwen2.5", # Use the clear and simple Qwen template
        )
        
        logger.info("✓ Model and tokenizer are ready for training.")
        return model, tokenizer

    def create_formatted_datasets(self, train_samples, val_samples, tokenizer):
        """Formats the data splits into Hugging Face Datasets."""
        def format_prompt(example):
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example['user_input']},
                {"role": "assistant", "content": json.dumps(example['ticket_data'])}
            ]
            return {"text": tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)}

        train_dataset = Dataset.from_list(train_samples).map(format_prompt)
        val_dataset = Dataset.from_list(val_samples).map(format_prompt)
        
        return DatasetDict({"train": train_dataset, "validation": val_dataset})

    def train(self, model, tokenizer, datasets):
        """Configures and runs the SFTTrainer."""
        logger.info("🚀 Starting model training...")
        
        # Correctly set FP16/BF16 flags
        use_fp16 = not is_bfloat16_supported()
        use_bf16 = is_bfloat16_supported()

        training_args = TrainingArguments(
            per_device_train_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
            warmup_steps=self.config["warmup_steps"],
            num_train_epochs=self.config["num_epochs"],
            learning_rate=self.config["learning_rate"],
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=self.config["logging_steps"],
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=self.config["output_dir"],
            # Evaluation and Saving Strategy
            eval_strategy="steps",
            eval_steps=self.config["eval_steps"],
            save_strategy="steps",
            save_steps=self.config["save_steps"],
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # W&B and Hub Integration
            report_to=self.config['report_to'],
            push_to_hub=bool(os.getenv('HF_TOKEN')),
            hub_model_id=self.config.get("hub_repo_id"),
            hub_strategy="every_save"
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            dataset_text_field="text",
            max_seq_length=self.config["max_seq_length"],
            args=training_args,
            packing=False, # Important for chat-formatted data
        )
        
        # Log GPU memory before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            start_gpu_mem = torch.cuda.memory_allocated() / 1e9
            logger.info(f"GPU Memory allocated before training: {start_gpu_mem:.3f} GB")

        trainer.train()

        # Log GPU memory after training
        if torch.cuda.is_available():
            end_gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"Peak GPU Memory during training: {end_gpu_mem:.3f} GB")

        logger.info("✓ Training complete. Saving final model.")
        model.save_pretrained(self.config["output_dir"])
        tokenizer.save_pretrained(self.config["output_dir"])
        
        return trainer

    def evaluate_and_log(self, model, tokenizer, test_samples: List[Dict]):
        """Evaluates the model on the test set and logs detailed metrics."""
        logger.info("\n" + "="*80)
        logger.info("🔍 Starting Final Evaluation on Test Set")
        logger.info("="*80)
        
        FastLanguageModel.for_inference(model)
        
        results = []
        parsing_failures = 0

        for i, sample in enumerate(test_samples):
            if (i + 1) % 20 == 0:
                logger.info(f"Evaluating sample {i+1}/{len(test_samples)}...")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["user_input"]},
            ]
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True, pad_token_id=tokenizer.eos_token_id)
            
            response_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
            
            predicted_data = None
            try:
                # Robust JSON extraction
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    predicted_data = json.loads(json_match.group(0))
                else:
                    raise json.JSONDecodeError("No JSON object found in response", response_text, 0)
            except json.JSONDecodeError:
                parsing_failures += 1
                predicted_data = {"error": "JSON Parsing Failed"}

            results.append({
                "user_input": sample["user_input"],
                "true_data": sample["ticket_data"],
                "predicted_data": predicted_data,
                "full_response": response_text
            })

        # Save raw results for inspection
        results_path = os.path.join(self.config["output_dir"], "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Full test results saved to {results_path}")
        
        # Calculate and log metrics
        self._calculate_and_display_metrics(results, parsing_failures)

    def _calculate_and_display_metrics(self, results, parsing_failures):
        """Helper to compute and log metrics from evaluation results."""
        total_samples = len(results)
        fields_to_evaluate = ["ticket_type", "severity", "department_impacted", "service_impacted"]
        metrics = {field: {"true": [], "pred": []} for field in fields_to_evaluate}
        
        for res in results:
            if "error" not in res["predicted_data"]:
                for field in fields_to_evaluate:
                    metrics[field]["true"].append(res["true_data"].get(field, "N/A"))
                    metrics[field]["pred"].append(res["predicted_data"].get(field, "Missing"))
        
        logger.info("\n" + "="*80)
        logger.info("📊 Final Performance Metrics")
        logger.info("="*80)

        parsing_success_rate = (total_samples - parsing_failures) / total_samples
        logger.info(f"JSON Parsing Success Rate: {parsing_success_rate:.2%} ({total_samples - parsing_failures}/{total_samples})")
        
        wandb_log = {"eval/parsing_success_rate": parsing_success_rate}
        
        for field, data in metrics.items():
            if not data["true"]: continue
            
            accuracy = accuracy_score(data["true"], data["pred"])
            report = classification_report(data["true"], data["pred"], zero_division=0, output_dict=True)
            
            logger.info(f"\n--- Metrics for: {field.upper()} ---")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"\nClassification Report:\n{classification_report(data['true'], data['pred'], zero_division=0)}")
            
            # Log to W&B
            wandb_log[f"eval/{field}_accuracy"] = accuracy
            wandb_log[f"eval/{field}_f1_weighted"] = report["weighted avg"]["f1-score"]

            # Plot and save confusion matrix
            self.plot_confusion_matrix(data["true"], data["pred"], field)

        if self.config['report_to'] == 'wandb':
            wandb.log(wandb_log)
            logger.info("✓ Metrics logged to W&B.")

    def plot_confusion_matrix(self, y_true, y_pred, field_name):
        """Plots and saves a confusion matrix for a given field."""
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix for: {field_name.upper()}', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = os.path.join(self.config["output_dir"], f"confusion_matrix_{field_name}.png")
        plt.savefig(cm_path)
        plt.close()
        
        logger.info(f"✓ Confusion matrix for '{field_name}' saved to {cm_path}")
        if self.config['report_to'] == 'wandb':
            wandb.log({f"eval/cm_{field_name}": wandb.Image(cm_path)})


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen-2.5 model for banking complaint classification.")
    parser.add_argument('--data_path', type=str, default='complaints.json', help='Path to the JSON dataset.')
    parser.add_argument('--output_dir', type=str, default='./qwen-complaint-agent', help='Directory to save the trained model and results.')
    parser.add_argument('--hub_repo_id', type=str, default='LaythAbuJafar/QwenInstruct7b_ComplaintAgent_Unsloth', help='Hugging Face Hub repository ID for model upload.')
    parser.add_argument('--model_name', type=str, default='unsloth/Qwen2.5-7B-Instruct-bnb-4bit', help='Base model from Hugging Face.')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size per device.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Initial learning rate.')
    parser.add_argument('--max_seq_length', type=int, default=2048, help='Maximum sequence length for the model.')
    
    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 4, # Effective batch size = batch_size * grad_accum_steps
        "num_epochs": args.num_epochs,
        "warmup_steps": 50,
        "eval_steps": 50,
        "save_steps": 50,
        "logging_steps": 5,
        "lora_r": 32,
        "lora_alpha": 64,
        "output_dir": args.output_dir,
        "hub_repo_id": args.hub_repo_id,
        "wandb_project": "banking-complaint-classifier",
        "report_to": "wandb" if os.getenv("WANDB_API_KEY") else "none"
    }
    
    logger.info(f"Starting training run with configuration:\n{json.dumps(config, indent=2)}")

    try:
        trainer_instance = ComplaintClassifierTrainer(config)
        trainer_instance.check_gpu_compatibility()
        
        train_samples, val_samples, test_samples = trainer_instance.load_and_prepare_data(args.data_path)
        
        model, tokenizer = trainer_instance.get_model_and_tokenizer()
        
        datasets = trainer_instance.create_formatted_datasets(train_samples, val_samples, tokenizer)
        
        trainer = trainer_instance.train(model, tokenizer, datasets)
        
        trainer_instance.evaluate_and_log(trainer.model, tokenizer, test_samples)
        
        if os.getenv("HF_TOKEN"):
            logger.info("🚀 Pushing final model to Hugging Face Hub...")
            trainer.push_to_hub()
            logger.info(f"✓ Model successfully pushed to {config['hub_repo_id']}")
        
        if wandb.run:
            wandb.finish()
            
        logger.info("🎉🎉🎉 Training and Evaluation Pipeline Finished Successfully! 🎉🎉🎉")

    except Exception as e:
        logger.error("❌ An unexpected error occurred during the pipeline.", exc_info=True)
        if wandb.run:
            wandb.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    main()