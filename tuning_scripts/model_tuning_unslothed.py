#!/usr/bin/env python3
"""
Pod-optimized training script for banking complaint classification with Unsloth
"""

import json
import torch
import pandas as pd
import os
import sys
import subprocess
import logging
import random
import numpy as np
import argparse
import re
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup comprehensive logging
def setup_logging():
    """Setup comprehensive logging for pod environment"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def install_dependencies():
    """Install required packages for pod environment"""
    logger.info("Installing dependencies...")
    
    packages = [
        "torch==2.1.0",
        "torchvision==0.16.0", 
        "torchaudio==2.1.0",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "xformers<0.0.22",
        "trl<0.9.0",
        "peft",
        "accelerate",
        "bitsandbytes",
        "transformers",
        "datasets",
        "wandb",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "huggingface_hub"
    ]
    
    for package in packages:
        try:
            if "unsloth" in package:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            elif "trl" in package or "xformers" in package:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--no-deps", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            else:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            logger.info(f"✓ Installed {package.split('@')[0] if '@' in package else package}")
        except Exception as e:
            logger.error(f"✗ Failed to install {package}: {e}")
            if "unsloth" in package or "torch" in package:
                raise Exception(f"Critical dependency failed: {package}")

def setup_environment():
    """Setup pod environment and authentication"""
    logger.info("Setting up environment...")
    
    # Create directories
    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    # Setup authentication
    wandb_key = os.getenv('WANDB_API_KEY')
    hf_token = os.getenv('HF_TOKEN')
    
    if wandb_key:
        try:
            import wandb
            wandb.login(key=wandb_key)
            logger.info("✓ W&B authentication successful")
        except Exception as e:
            logger.warning(f"W&B authentication failed: {e}")
    else:
        logger.warning("⚠ WANDB_API_KEY not found - experiment tracking disabled")
    
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token)
            logger.info("✓ HF authentication successful")
        except Exception as e:
            logger.warning(f"HF authentication failed: {e}")
    else:
        logger.warning("⚠ HF_TOKEN not found - model upload will fail")

# Install dependencies first
logger.info("🚀 Starting pod setup...")
install_dependencies()
setup_environment()

# Now import the ML libraries
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
    from trl import SFTTrainer, SFTConfig
    logger.info("✓ All ML libraries imported successfully")
except ImportError as e:
    logger.error(f"Failed to import ML libraries: {e}")
    raise

# Configuration
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

class UnslothComplaintTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_reproducibility()
        self.setup_wandb()
        
    def setup_reproducibility(self):
        """Set seeds for reproducibility"""
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
    def setup_wandb(self):
        """Setup W&B with error handling"""
        try:
            wandb.init(
                project="banking-complaint-classifier-pod",
                name=f"qwen-unsloth-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config,
                resume="allow"
            )
            logger.info("✓ W&B initialized successfully")
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")

    def check_gpu_compatibility(self):
        """Check GPU compatibility and memory"""
        if not torch.cuda.is_available():
            raise SystemExit("❌ CUDA not available. This script requires a GPU.")
        
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"✓ GPU: {gpu_name}")
        logger.info(f"✓ CUDA version: {torch.version.cuda}")
        logger.info(f"✓ GPU memory: {gpu_memory:.1f} GB")
        logger.info(f"✓ Unsloth bfloat16 support: {is_bfloat16_supported()}")
        
        if gpu_memory < 15:
            logger.warning("⚠ GPU memory < 15GB - consider reducing batch size")

    def load_data_flexible(self, data_source: str) -> List[Dict]:
        """Load data from various sources"""
        logger.info(f"Loading data from: {data_source}")
        
        try:
            if data_source.startswith('http'):
                import requests
                response = requests.get(data_source)
                response.raise_for_status()
                data = response.json()
                logger.info(f"✓ Downloaded data from URL: {len(data)} samples")
            elif data_source.startswith('hf://'):
                from datasets import load_dataset
                dataset_name = data_source.replace('hf://', '')
                dataset = load_dataset(dataset_name)
                data = dataset['train'].to_list()
                logger.info(f"✓ Loaded from HF dataset: {len(data)} samples")
            else:
                with open(data_source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"✓ Loaded local file: {len(data)} samples")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def analyze_data_quality(self, data: List[Dict]):
        """Comprehensive data quality analysis"""
        logger.info("=== DATA QUALITY ANALYSIS ===")
        
        # Basic statistics
        logger.info(f"Total samples: {len(data)}")
        
        # Remove duplicates
        unique_data = []
        seen_inputs = set()
        for sample in data:
            if sample['user_input'] not in seen_inputs:
                unique_data.append(sample)
                seen_inputs.add(sample['user_input'])
        
        duplicate_count = len(data) - len(unique_data)
        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicates ({duplicate_count/len(data)*100:.1f}%)")
        
        # Distribution analysis
        ticket_types = [sample['ticket_data']['ticket_type'] for sample in unique_data]
        type_counts = pd.Series(ticket_types).value_counts()
        logger.info("Ticket Type Distribution:")
        for ticket_type, count in type_counts.items():
            logger.info(f"  {ticket_type}: {count} ({count/len(unique_data)*100:.1f}%)")
        
        # Diversity checks
        departments = [sample['ticket_data']['department_impacted'] for sample in unique_data]
        services = [sample['ticket_data']['service_impacted'] for sample in unique_data]
        
        logger.info(f"Department diversity: {len(set(departments))} unique")
        logger.info(f"Service diversity: {len(set(services))} unique")
        
        # Text length analysis
        text_lengths = [len(sample['user_input']) for sample in unique_data]
        logger.info(f"Text length - Mean: {np.mean(text_lengths):.1f}, "
                   f"Median: {np.median(text_lengths):.1f}, "
                   f"Max: {max(text_lengths)}")
        
        return unique_data

    def create_stratified_splits(self, samples: List[Dict], 
                               train_ratio: float = 0.7, 
                               val_ratio: float = 0.15):
        """Create stratified train/val/test splits"""
        logger.info("=== CREATING STRATIFIED SPLITS ===")
        
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
            
            logger.info(f"{ticket_type}: {len(group_samples[:train_end])} train, "
                       f"{len(group_samples[train_end:val_end])} val, "
                       f"{len(group_samples[val_end:])} test")
        
        # Shuffle final splits
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        logger.info(f"Final splits - Train: {len(train_samples)}, "
                   f"Val: {len(val_samples)}, Test: {len(test_samples)}")
        
        return train_samples, val_samples, test_samples

    def prepare_model_and_tokenizer(self):
        """Prepare model with Unsloth optimization"""
        logger.info("🚀 Preparing model with Unsloth...")
        
        try:
            # Load model with Unsloth
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config["model_name"],
                max_seq_length=self.config["max_length"],
                dtype=None,  # Auto-detect
                load_in_4bit=True,
            )
            
            # Apply LoRA
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.config["lora_r"],
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_alpha=self.config["lora_alpha"],
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
            
            # Setup chat template
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="qwen-2.5",
            )
            
            logger.info("✓ Model and tokenizer prepared successfully")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to prepare model: {e}")
            raise

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

    def create_datasets(self, train_samples: List[Dict], 
                       val_samples: List[Dict], 
                       test_samples: List[Dict], 
                       tokenizer):
        """Create datasets with proper formatting"""
        logger.info("Creating datasets...")
        
        def format_dataset(samples_list):
            formatted_data = []
            for sample in samples_list:
                try:
                    text = self.create_training_prompt(sample, tokenizer)
                    formatted_data.append({"text": text})
                except Exception as e:
                    logger.warning(f"Skipped sample due to formatting error: {e}")
            return Dataset.from_list(formatted_data)
        
        dataset_dict = DatasetDict({
            "train": format_dataset(train_samples),
            "validation": format_dataset(val_samples)
        })
        
        logger.info(f"✓ Created datasets - Train: {len(dataset_dict['train'])}, "
                   f"Val: {len(dataset_dict['validation'])}")
        
        return dataset_dict, test_samples

    def train_model(self, model, tokenizer, datasets):
        """Train model with Unsloth optimization"""
        logger.info("🚀 Starting training...")
        
        # Memory stats before training
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logger.info(f"GPU: {gpu_stats.name}, Max memory: {max_memory} GB")
            logger.info(f"Memory reserved before training: {start_gpu_memory} GB")
        
        try:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=datasets["train"],
                eval_dataset=datasets["validation"],
                dataset_text_field="text",
                max_seq_length=self.config["max_length"],
                dataset_num_proc=2,
                packing=False,
                args=SFTConfig(
                    per_device_train_batch_size=self.config["batch_size"],
                    per_device_eval_batch_size=self.config["batch_size"],
                    gradient_accumulation_steps=self.config["gradient_accumulation_steps"],
                    warmup_steps=self.config["warmup_steps"],
                    num_train_epochs=self.config["num_epochs"],
                    learning_rate=self.config["learning_rate"],
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=self.config["logging_steps"],
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="cosine",
                    seed=3407,
                    output_dir=self.config["output_dir"],
                    eval_strategy="steps",
                    eval_steps=self.config["eval_steps"],
                    save_strategy="steps",
                    save_steps=self.config["save_steps"],
                    save_total_limit=3,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    report_to="wandb" if wandb.run else "none",
                    push_to_hub=bool(os.getenv('HF_TOKEN')),
                    hub_model_id=self.config.get("hub_repo"),
                    hub_strategy="every_save" if os.getenv('HF_TOKEN') else "end",
                    dataloader_num_workers=2,
                    remove_unused_columns=False,
                ),
            )
            
            # Train
            trainer_stats = trainer.train()
            
            # Memory stats after training
            if torch.cuda.is_available():
                used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
                used_memory_for_training = round(used_memory - start_gpu_memory, 3)
                logger.info(f"Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds")
                logger.info(f"Peak memory usage: {used_memory} GB")
                logger.info(f"Memory used for training: {used_memory_for_training} GB")
            
            # Save model locally
            model.save_pretrained(self.config["output_dir"])
            tokenizer.save_pretrained(self.config["output_dir"])
            logger.info(f"✓ Model saved to {self.config['output_dir']}")
            
            return trainer
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def evaluate_model(self, model, tokenizer, test_samples: List[Dict]):
        """Comprehensive model evaluation"""
        logger.info("=== STARTING EVALUATION ===")
        
        # Switch to inference mode
        FastLanguageModel.for_inference(model)
        
        metrics = {
            'ticket_type': {'pred': [], 'true': []},
            'severity': {'pred': [], 'true': []},
            'department': {'pred': [], 'true': []},
            'service': {'pred': [], 'true': []}
        }
        
        successful_predictions = 0
        total_samples = len(test_samples)
        
        logger.info(f"Evaluating on {total_samples} test samples...")
        
        for i, sample in enumerate(test_samples):
            if i % 50 == 0:
                logger.info(f"Progress: {i+1}/{total_samples}")
            
            try:
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
                
                # Generate response
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
                
                # Decode and parse response
                response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
                
                # Extract JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    pred_ticket = json.loads(json_match.group())
                    true_data = sample["ticket_data"]
                    
                    # Store predictions
                    metrics['ticket_type']['pred'].append(pred_ticket.get("ticket_type", "unknown"))
                    metrics['ticket_type']['true'].append(true_data.get("ticket_type", "unknown"))
                    
                    metrics['severity']['pred'].append(pred_ticket.get("severity", "unknown"))
                    metrics['severity']['true'].append(true_data.get("severity", "unknown"))
                    
                    metrics['department']['pred'].append(pred_ticket.get("department_impacted", "unknown"))
                    metrics['department']['true'].append(true_data.get("department_impacted", "unknown"))
                    
                    metrics['service']['pred'].append(pred_ticket.get("service_impacted", "unknown"))
                    metrics['service']['true'].append(true_data.get("service_impacted", "unknown"))
                    
                    successful_predictions += 1
                else:
                    # Handle parsing failures
                    for field in metrics:
                        metrics[field]['pred'].append("parse_error")
                        metrics[field]['true'].append(sample["ticket_data"].get(
                            field if field != 'department' else 'department_impacted',
                            field if field != 'service' else 'service_impacted'
                        ))
                        
            except Exception as e:
                logger.warning(f"Evaluation error for sample {i}: {e}")
                # Add error entries
                for field in metrics:
                    metrics[field]['pred'].append("error")
                    metrics[field]['true'].append(sample["ticket_data"].get(
                        field if field != 'department' else 'department_impacted',
                        field if field != 'service' else 'service_impacted'
                    ))
        
        # Calculate and log metrics
        self.log_evaluation_results(metrics, successful_predictions, total_samples)
        
        return metrics

    def log_evaluation_results(self, metrics: Dict, successful_predictions: int, total_samples: int):
        """Log comprehensive evaluation results"""
        logger.info("=" * 80)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 80)
        
        # Overall success rate
        success_rate = successful_predictions / total_samples
        logger.info(f"Successful predictions: {successful_predictions}/{total_samples} ({success_rate:.2%})")
        
        # Field-specific metrics
        evaluation_results = {}
        for field_name, field_data in metrics.items():
            if len(field_data['true']) > 0:
                accuracy = accuracy_score(field_data['true'], field_data['pred'])
                precision, recall, f1, _ = precision_recall_fscore_support(
                    field_data['true'], field_data['pred'], average='weighted', zero_division=0
                )
                
                logger.info(f"\n{field_name.upper()}:")
                logger.info(f"  Accuracy: {accuracy:.4f}")
                logger.info(f"  Precision: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
                
                evaluation_results[f"{field_name}_accuracy"] = accuracy
                evaluation_results[f"{field_name}_precision"] = precision
                evaluation_results[f"{field_name}_recall"] = recall
                evaluation_results[f"{field_name}_f1"] = f1
        
        # Log to W&B
        if wandb.run:
            wandb.log({
                "eval_success_rate": success_rate,
                **evaluation_results
            })
        
        # Save detailed results
        results_path = os.path.join(self.config["output_dir"], "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                "success_rate": success_rate,
                "successful_predictions": successful_predictions,
                "total_samples": total_samples,
                "field_metrics": evaluation_results,
                "detailed_metrics": metrics
            }, f, indent=2)
        
        logger.info(f"✓ Detailed results saved to {results_path}")

    def run_complete_pipeline(self, data_source: str):
        """Run the complete training and evaluation pipeline"""
        logger.info("🚀 STARTING COMPLETE TRAINING PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Setup and checks
            self.check_gpu_compatibility()
            
            # Load and analyze data
            raw_data = self.load_data_flexible(data_source)
            clean_data = self.analyze_data_quality(raw_data)
            
            # Create splits
            train_samples, val_samples, test_samples = self.create_stratified_splits(clean_data)
            
            # Prepare model
            model, tokenizer = self.prepare_model_and_tokenizer()
            
            # Create datasets
            datasets, test_samples = self.create_datasets(train_samples, val_samples, test_samples, tokenizer)
            
            # Train model
            trainer = self.train_model(model, tokenizer, datasets)
            
            # Evaluate model
            metrics = self.evaluate_model(trainer.model, tokenizer, test_samples)
            
            # Upload to Hub if token available
            if os.getenv('HF_TOKEN') and self.config.get("hub_repo"):
                try:
                    trainer.push_to_hub()
                    logger.info(f"✓ Model uploaded to {self.config['hub_repo']}")
                except Exception as e:
                    logger.warning(f"Hub upload failed: {e}")
            
            # Cleanup
            if wandb.run:
                wandb.finish()
            
            logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            return trainer, metrics
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if wandb.run:
                wandb.finish()
            raise

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train banking complaint classifier with Unsloth')
    parser.add_argument('--data_path', type=str, default='training_data_fixed.json',
                       help='Path to training data (local file, URL, or hf://dataset_name)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for model and results')
    parser.add_argument('--hub_repo', type=str, default='LaythAbuJafar/QwenInstruct7b_ComplaintAgent_Unsloth',
                       help='Hugging Face repository for model upload')
    parser.add_argument('--model_name', type=str, default='unsloth/Qwen2.5-7B-Instruct-bnb-4bit',
                       help='Base model name')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 2,
        "num_epochs": args.epochs,
        "warmup_steps": 100,
        "eval_steps": 50,
        "save_steps": 100,
        "logging_steps": 10,
        "lora_r": 64,
        "lora_alpha": 128,
        "output_dir": args.output_dir,
        "hub_repo": args.hub_repo,
        "unsloth_optimized": True
    }
    
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    try:
        # Initialize trainer
        trainer_instance = UnslothComplaintTrainer(config)
        
        # Run pipeline
        trainer, metrics = trainer_instance.run_complete_pipeline(args.data_path)
        
        logger.info("✅ Training completed successfully!")
        return trainer, metrics
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return None, None
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
