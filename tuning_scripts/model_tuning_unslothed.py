import json
import torch
import pandas as pd
from typing import Dict, List, Tuple
import re
import os
import argparse
import random
import numpy as np
from datasets import Dataset, DatasetDict

# Unsloth imports - replace standard transformers
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
from datetime import datetime
from trl import SFTTrainer, SFTConfig
import warnings
warnings.filterwarnings('ignore')

# --- Enhanced Configuration with Unsloth ---
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"  # Unsloth pre-quantized model
HF_HUB_REPO_ID = "LaythAbuJafar/QwenInstruct7b_ComplaintAgent"
OUTPUT_DIR = "./banking-ticket-model-unsloth"
MAX_LENGTH = 2048
LEARNING_RATE = 2e-4
BATCH_SIZE = 8  # Increased due to lower memory usage
GRADIENT_ACCUMULATION_STEPS = 2  # Reduced due to larger batch size
NUM_EPOCHS = 3
WARMUP_STEPS = 100
EVAL_STEPS = 50
SAVE_STEPS = 100
LOGGING_STEPS = 10

# Enhanced system prompt
SYSTEM_PROMPT = """You are a banking customer service ticket classification and filling assistant. Your role is to:

1. Analyze customer inputs and extract relevant information.
2. Fill ticket fields accurately based on the customer's request.
3. Stay strictly within the banking and financial services domain.
4. Reject any requests outside of banking support.

You must ONLY respond with a valid JSON object containing the ticket fields. Do not provide any other information or engage in conversation.

Required fields:
- ticket_type: "complaint", "inquiry", or "assistance"
- title: Brief summary of the issue
- description: Detailed description
- severity: "low", "medium", "high", or "critical"
- department_impacted: The bank department affected
- service_impacted: The specific service affected
- preferred_communication: "email", "phone", "chat", "in-person", or ""
- assistance_request: (ONLY if ticket_type is "assistance") - specific assistance needed

If the request is not related to banking, respond with: {"error": "Request outside banking domain"}"""

class UnslothEnhancedTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.setup_reproducibility()
        
    def setup_logging(self):
        """Setup comprehensive logging with W&B"""
        wandb.init(
            project="banking-complaint-classifier-unsloth",
            name=f"qwen-unsloth-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=self.config
        )
        
    def setup_reproducibility(self):
        """Set seeds for reproducibility"""
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    def check_gpu_compatibility(self):
        if not torch.cuda.is_available():
            print("❌ CRITICAL: CUDA not available. This script requires a GPU.")
            raise SystemExit("CUDA is not available. Please check your installation.")
        
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"✓ Unsloth bfloat16 support: {is_bfloat16_supported()}")

    def prepare_model_and_tokenizer_unsloth(self):
        """Prepare model with Unsloth optimization"""
        print("🚀 Preparing model with Unsloth optimization...")
        
        # Load model with Unsloth - automatically handles quantization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_LENGTH,
            dtype=None,  # Auto-detect optimal dtype
            load_in_4bit=True,
            # token="hf_...", # Use if model is gated
        )
        
        # Apply Unsloth optimizations with LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=64,  # LoRA rank
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_alpha=128,
            lora_dropout=0,  # Unsloth optimizes for 0 dropout
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
            random_state=3407,
            use_rslora=False,  # Rank stabilized LoRA
            loftq_config=None,  # LoftQ quantization
        )
        
        # Setup chat template for Qwen
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="qwen-2.5",  # Use Qwen's chat template
        )
        
        return model, tokenizer

    def create_training_prompt(self, example: Dict, tokenizer) -> str:
        """Create training prompt using Qwen chat template"""
        user_input = example['user_input']
        ticket_data = example['ticket_data']
        
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": json.dumps(ticket_data, indent=2)}
        ]
        
        return tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )

    def load_and_analyze_data(self, json_path: str) -> List[Dict]:
        """Load data with comprehensive analysis"""
        print(f"Loading data from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples.")
        
        # Data quality analysis
        self.analyze_data_quality(data)
        
        # Remove duplicates based on user_input
        unique_data = []
        seen_inputs = set()
        for sample in data:
            if sample['user_input'] not in seen_inputs:
                unique_data.append(sample)
                seen_inputs.add(sample['user_input'])
        
        print(f"After deduplication: {len(unique_data)} unique samples")
        return unique_data

    def analyze_data_quality(self, data: List[Dict]):
        """Comprehensive data quality analysis"""
        print("\n=== DATA QUALITY ANALYSIS ===")
        
        # Basic statistics
        print(f"Total samples: {len(data)}")
        
        # Ticket type distribution
        ticket_types = [sample['ticket_data']['ticket_type'] for sample in data]
        type_counts = pd.Series(ticket_types).value_counts()
        print(f"\nTicket Type Distribution:")
        for ticket_type, count in type_counts.items():
            print(f"  {ticket_type}: {count} ({count/len(data)*100:.1f}%)")
        
        # Department diversity check
        departments = [sample['ticket_data']['department_impacted'] for sample in data]
        dept_counts = pd.Series(departments).value_counts()
        print(f"\nDepartment Diversity: {len(dept_counts)} unique departments")
        
        # Service diversity check
        services = [sample['ticket_data']['service_impacted'] for sample in data]
        service_counts = pd.Series(services).value_counts()
        print(f"Service Diversity: {len(service_counts)} unique services")

    def create_stratified_splits(self, samples: List[Dict], train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Create stratified train/val/test splits"""
        print(f"\n=== CREATING STRATIFIED SPLITS ===")
        
        # Group by ticket_type for stratification
        type_groups = {}
        for sample in samples:
            ticket_type = sample['ticket_data']['ticket_type']
            if ticket_type not in type_groups:
                type_groups[ticket_type] = []
            type_groups[ticket_type].append(sample)
        
        train_samples, val_samples, test_samples = [], [], []
        
        for ticket_type, group_samples in type_groups.items():
            random.shuffle(group_samples)
            n_samples = len(group_samples)
            
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            train_samples.extend(group_samples[:train_end])
            val_samples.extend(group_samples[train_end:val_end])
            test_samples.extend(group_samples[val_end:])
            
            print(f"{ticket_type}: {len(group_samples[:train_end])} train, "
                  f"{len(group_samples[train_end:val_end])} val, "
                  f"{len(group_samples[val_end:])} test")
        
        # Shuffle the final splits
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        print(f"\nFinal splits - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        
        return train_samples, val_samples, test_samples

    def create_datasets(self, train_samples: List[Dict], val_samples: List[Dict], 
                       test_samples: List[Dict], tokenizer):
        """Create datasets with proper formatting"""
        def format_dataset(samples_list):
            return Dataset.from_list([
                {"text": self.create_training_prompt(s, tokenizer)} 
                for s in samples_list
            ])
        
        dataset_dict = DatasetDict({
            "train": format_dataset(train_samples),
            "validation": format_dataset(val_samples)
        })
        
        return dataset_dict, test_samples

    def train_model_unsloth(self, model, tokenizer, datasets):
        """Enhanced training with Unsloth optimization"""
        print("🚀 Starting Unsloth-optimized training...")
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            dataset_text_field="text",
            max_seq_length=MAX_LENGTH,
            dataset_num_proc=2,
            packing=False,  # Can cause issues with chat templates
            args=SFTConfig(
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                warmup_steps=WARMUP_STEPS,
                num_train_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=LOGGING_STEPS,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=3407,
                output_dir=OUTPUT_DIR,
                eval_strategy="steps",
                eval_steps=EVAL_STEPS,
                save_strategy="steps",
                save_steps=SAVE_STEPS,
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="wandb",
                push_to_hub=True,
                hub_model_id=HF_HUB_REPO_ID,
                hub_strategy="every_save",
                dataloader_num_workers=2,
                remove_unused_columns=False,
            ),
        )
        
        # Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
        
        trainer_stats = trainer.train()
        
        # Final memory and time stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory         /max_memory*100, 3)
        lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        
        # Save model
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        return trainer

    def comprehensive_evaluation(self, model, tokenizer, test_samples: List[Dict]):
        """Comprehensive model evaluation with detailed metrics"""
        print("\n=== COMPREHENSIVE MODEL EVALUATION ===")
        
        # Switch to evaluation mode
        FastLanguageModel.for_inference(model)
        
        predictions = []
        true_labels = []
        prediction_details = []
        
        # Evaluation metrics storage
        metrics = {
            'ticket_type': {'pred': [], 'true': []},
            'severity': {'pred': [], 'true': []},
            'department': {'pred': [], 'true': []},
            'service': {'pred': [], 'true': []}
        }
        
        print(f"Evaluating on {len(test_samples)} test samples...")
        
        for i, sample in enumerate(test_samples):
            if i % 50 == 0:
                print(f"Processing sample {i+1}/{len(test_samples)}")
                
            if "error" in sample["ticket_data"]:
                continue
                
            # Create prompt
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["user_input"]},
            ]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=True
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
            
            # Parse JSON response
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    pred_ticket = json.loads(json_match.group())
                    
                    # Store predictions for each field
                    true_data = sample["ticket_data"]
                    
                    metrics['ticket_type']['pred'].append(pred_ticket.get("ticket_type", "unknown"))
                    metrics['ticket_type']['true'].append(true_data.get("ticket_type", "unknown"))
                    
                    metrics['severity']['pred'].append(pred_ticket.get("severity", "unknown"))
                    metrics['severity']['true'].append(true_data.get("severity", "unknown"))
                    
                    metrics['department']['pred'].append(pred_ticket.get("department_impacted", "unknown"))
                    metrics['department']['true'].append(true_data.get("department_impacted", "unknown"))
                    
                    metrics['service']['pred'].append(pred_ticket.get("service_impacted", "unknown"))
                    metrics['service']['true'].append(true_data.get("service_impacted", "unknown"))
                    
                    prediction_details.append({
                        'input': sample["user_input"],
                        'true': true_data,
                        'pred': pred_ticket,
                        'correct': pred_ticket.get("ticket_type") == true_data.get("ticket_type")
                    })
                else:
                    # Handle parsing failures
                    for field in metrics:
                        metrics[field]['pred'].append("parse_error")
                        metrics[field]['true'].append(sample["ticket_data"].get(
                            field if field != 'department' else 'department_impacted',
                            field if field != 'service' else 'service_impacted'
                        ))
                    
            except json.JSONDecodeError:
                # Handle JSON decode errors
                for field in metrics:
                    metrics[field]['pred'].append("json_error")
                    metrics[field]['true'].append(sample["ticket_data"].get(
                        field if field != 'department' else 'department_impacted',
                        field if field != 'service' else 'service_impacted'
                    ))
        
        # Calculate and display metrics
        self.display_comprehensive_metrics(metrics, prediction_details)
        
        return metrics, prediction_details

    def display_comprehensive_metrics(self, metrics: Dict, prediction_details: List[Dict]):
        """Display comprehensive evaluation metrics"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        
        # Overall accuracy
        overall_correct = sum(1 for detail in prediction_details if detail['correct'])
        overall_accuracy = overall_correct / len(prediction_details) if prediction_details else 0
        print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_correct}/{len(prediction_details)})")
        
        # Field-specific metrics
        for field_name, field_data in metrics.items():
            print(f"\n{field_name.upper()} CLASSIFICATION:")
            print("-" * 40)
            
            accuracy = accuracy_score(field_data['true'], field_data['pred'])
            print(f"Accuracy: {accuracy:.4f}")
            
            # Precision, Recall, F1
            precision, recall, f1, support = precision_recall_fscore_support(
                field_data['true'], field_data['pred'], average='weighted', zero_division=0
            )
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
        
        # Log metrics to W&B
        wandb.log({
            "test_overall_accuracy": overall_accuracy,
            "test_ticket_type_accuracy": accuracy_score(metrics['ticket_type']['true'], metrics['ticket_type']['pred']),
            "test_severity_accuracy": accuracy_score(metrics['severity']['true'], metrics['severity']['pred']),
            "test_department_accuracy": accuracy_score(metrics['department']['true'], metrics['department']['pred']),
            "test_service_accuracy": accuracy_score(metrics['service']['true'], metrics['service']['pred'])
        })

    def run_training_pipeline(self, json_path: str):
        """Run the complete training pipeline with Unsloth"""
        print("Starting Unsloth-Enhanced Training Pipeline")
        print("="*80)
        
        # Setup
        self.check_gpu_compatibility()
        
        # Load and analyze data
        samples = self.load_and_analyze_data(json_path)
        
        # Create stratified splits
        train_samples, val_samples, test_samples = self.create_stratified_splits(samples)
        
        # Prepare model and tokenizer with Unsloth
        model, tokenizer = self.prepare_model_and_tokenizer_unsloth()
        
        # Create datasets
        datasets, test_samples = self.create_datasets(train_samples, val_samples, test_samples, tokenizer)
        
        # Train model with Unsloth optimization
        trainer = self.train_model_unsloth(model, tokenizer, datasets)
        
        # Comprehensive evaluation
        metrics, prediction_details = self.comprehensive_evaluation(trainer.model, tokenizer, test_samples)
        
        # Upload to Hub
        print("\nUploading final model and tokenizer to Hugging Face Hub...")
        trainer.push_to_hub()
        print(f"Successfully uploaded model to {HF_HUB_REPO_ID}")
        
        # Finish W&B run
        wandb.finish()
        
        print("\nUnsloth-Enhanced Training Pipeline Completed Successfully!")
        return trainer, metrics

def main():
    config = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lora_r": 64,
        "lora_alpha": 128,
        "unsloth_optimized": True
    }
    
    trainer_instance = UnslothEnhancedTrainer(config)
    
    # Run with your corrected dataset
    json_path = "datasets/complaints.json"
    trainer, metrics = trainer_instance.run_training_pipeline(json_path)
    
    return trainer, metrics

if __name__ == "__main__":
    main()
