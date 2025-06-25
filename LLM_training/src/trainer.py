"""
Training utilities for LoRA fine-tuning.

This module handles the training loop, monitoring, and evaluation
for LoRA-based fine-tuning of language models.
"""

import logging
import os
import math
from typing import Dict, Any, Optional, List
import torch
import wandb
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset
from peft import PeftModel
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    Custom trainer class for LoRA fine-tuning with enhanced monitoring.
    """
    
    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.training_config = config.get("training", {})
        self.wandb_config = config.get("wandb", {})
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize WandB if configured
        self.setup_wandb()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    
    def setup_wandb(self):
        """Initialize Weights & Biases logging if configured."""
        if "wandb" in self.training_config.get("report_to", []):
            try:
                wandb.init(
                    project=self.wandb_config.get("project", "lora-peft-llm"),
                    entity=self.wandb_config.get("entity"),
                    name=self.training_config.get("run_name", "lora-experiment"),
                    tags=self.wandb_config.get("tags", []),
                    notes=self.wandb_config.get("notes", ""),
                    config=self.config,
                )
                logger.info("WandB initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
    
    def create_training_arguments(self) -> TrainingArguments:
        """
        Create TrainingArguments from configuration.
        
        Returns:
            TrainingArguments object
        """
        # Extract training config
        args = {
            "output_dir": self.training_config.get("output_dir", "./outputs"),
            "overwrite_output_dir": self.training_config.get("overwrite_output_dir", True),
            "num_train_epochs": self.training_config.get("num_train_epochs", 3),
            "per_device_train_batch_size": self.training_config.get("per_device_train_batch_size", 4),
            "per_device_eval_batch_size": self.training_config.get("per_device_eval_batch_size", 4),
            "gradient_accumulation_steps": self.training_config.get("gradient_accumulation_steps", 4),
            "eval_accumulation_steps": self.training_config.get("eval_accumulation_steps", 1),
            
            # Learning rate and optimization
            "learning_rate": self.training_config.get("learning_rate", 3e-4),
            "weight_decay": self.training_config.get("weight_decay", 0.01),
            "adam_beta1": self.training_config.get("adam_beta1", 0.9),
            "adam_beta2": self.training_config.get("adam_beta2", 0.999),
            "adam_epsilon": self.training_config.get("adam_epsilon", 1e-8),
            "lr_scheduler_type": self.training_config.get("lr_scheduler_type", "cosine"),
            "warmup_steps": self.training_config.get("warmup_steps", 100),
            
            # Memory and performance
            "fp16": self.training_config.get("fp16", False),
            "bf16": self.training_config.get("bf16", True),
            "gradient_checkpointing": self.training_config.get("gradient_checkpointing", True),
            "dataloader_num_workers": self.training_config.get("dataloader_num_workers", 4),
            "remove_unused_columns": self.training_config.get("remove_unused_columns", False),
            
            # Evaluation and logging
            "evaluation_strategy": self.training_config.get("evaluation_strategy", "steps"),
            "eval_steps": self.training_config.get("eval_steps", 500),
            "save_strategy": self.training_config.get("save_strategy", "steps"),
            "save_steps": self.training_config.get("save_steps", 500),
            "logging_steps": self.training_config.get("logging_steps", 100),
            "load_best_model_at_end": self.training_config.get("load_best_model_at_end", True),
            "metric_for_best_model": self.training_config.get("metric_for_best_model", "eval_loss"),
            "greater_is_better": self.training_config.get("greater_is_better", False),
            
            # Miscellaneous
            "seed": self.training_config.get("seed", 42),
            "data_seed": self.training_config.get("data_seed", 42),
            "report_to": self.training_config.get("report_to", ["tensorboard"]),
            "run_name": self.training_config.get("run_name", "lora-llm-finetune"),
            
            # Advanced options
            "save_total_limit": 3,  # Keep only 3 checkpoints
            "prediction_loss_only": False,
            "include_inputs_for_metrics": False,
        }
        
        return TrainingArguments(**args)
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute custom metrics for evaluation.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            Dictionary of computed metrics
        """
        predictions, labels = eval_pred
        
        # Handle logits (predictions might be logits)
        if len(predictions.shape) == 3:
            # predictions shape: (batch_size, seq_len, vocab_size)
            predictions = np.argmax(predictions, axis=-1)
        
        # Flatten arrays and remove ignored tokens (-100)
        labels = labels.reshape(-1)
        predictions = predictions.reshape(-1)
        
        # Remove ignored tokens
        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]
        
        if len(labels) == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )
        
        # Compute perplexity from loss
        try:
            eval_loss = eval_pred.metrics.get("eval_loss", 0)
            perplexity = math.exp(eval_loss) if eval_loss < 100 else float("inf")
        except:
            perplexity = float("inf")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "perplexity": perplexity,
        }
    
    def create_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator=None,
    ) -> Trainer:
        """
        Create Trainer instance with all configurations.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator for batching
            
        Returns:
            Configured Trainer instance
        """
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Setup callbacks
        callbacks = []
        
        # Early stopping callback
        if self.training_config.get("early_stopping_patience"):
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.training_config.get("early_stopping_patience", 3),
                early_stopping_threshold=self.training_config.get("early_stopping_threshold", 0.001),
            )
            callbacks.append(early_stopping)
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            callbacks=callbacks,
        )
        
        return trainer
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator=None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main training function.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            data_collator: Data collator
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training results
        """
        logger.info("Starting LoRA training...")
        
        # Log model info
        self.log_model_info()
        
        # Create trainer
        trainer = self.create_trainer(train_dataset, eval_dataset, data_collator)
        
        # Check for existing checkpoints
        if resume_from_checkpoint is None:
            last_checkpoint = get_last_checkpoint(self.training_config.get("output_dir", "./outputs"))
            if last_checkpoint:
                logger.info(f"Found checkpoint at {last_checkpoint}")
                resume_from_checkpoint = last_checkpoint
        
        # Start training
        try:
            if resume_from_checkpoint:
                logger.info(f"Resuming training from {resume_from_checkpoint}")
                train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            else:
                train_result = trainer.train()
            
            # Save the final model
            logger.info("Saving final model...")
            trainer.save_model()
            trainer.save_state()
            
            # Log training results
            self.log_training_results(train_result)
            
            return train_result.metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Cleanup WandB
            if "wandb" in self.training_config.get("report_to", []):
                wandb.finish()
    
    def log_model_info(self):
        """Log information about the model being trained."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model: {type(self.model).__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
        # Log to WandB if available
        if "wandb" in self.training_config.get("report_to", []):
            wandb.log({
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/trainable_percentage": 100 * trainable_params / total_params,
            })
    
    def log_training_results(self, train_result):
        """Log final training results."""
        metrics = train_result.metrics
        
        logger.info("Training completed!")
        logger.info(f"Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        logger.info(f"Training time: {metrics.get('train_runtime', 'N/A'):.2f} seconds")
        logger.info(f"Samples per second: {metrics.get('train_samples_per_second', 'N/A'):.2f}")
        
        # Log to WandB if available
        if "wandb" in self.training_config.get("report_to", []):
            wandb.log({
                "final/train_loss": metrics.get("train_loss", 0),
                "final/train_runtime": metrics.get("train_runtime", 0),
                "final/train_samples_per_second": metrics.get("train_samples_per_second", 0),
            })
    
    def evaluate(
        self,
        eval_dataset: Dataset,
        data_collator=None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            eval_dataset: Dataset to evaluate on
            data_collator: Data collator
            
        Returns:
            Evaluation metrics
        """
        logger.info("Running evaluation...")
        
        # Create trainer for evaluation
        trainer = self.create_trainer(
            train_dataset=eval_dataset,  # Dummy train dataset
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Run evaluation
        eval_results = trainer.evaluate()
        
        logger.info("Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return eval_results


class MemoryCallback:
    """Callback to monitor GPU memory usage during training."""
    
    def __init__(self, log_every_n_steps: int = 100):
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Log memory usage at the end of each step."""
        self.step_count += 1
        
        if self.step_count % self.log_every_n_steps == 0:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                
                logger.info(f"Step {self.step_count}: GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                
                # Log to WandB if available
                if wandb.run:
                    wandb.log({
                        "memory/allocated_gb": memory_allocated,
                        "memory/reserved_gb": memory_reserved,
                        "step": self.step_count,
                    })


def print_training_summary(config: Dict[str, Any]):
    """
    Print a summary of the training configuration.
    
    Args:
        config: Training configuration
    """
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    
    # Model info
    model_config = config.get("model", {})
    print(f"Model: {model_config.get('model_name', 'Unknown')}")
    
    # LoRA info
    lora_config = config.get("lora", {})
    print(f"LoRA Rank (r): {lora_config.get('r', 16)}")
    print(f"LoRA Alpha: {lora_config.get('lora_alpha', 32)}")
    print(f"Target Modules: {lora_config.get('target_modules', 'Auto-detect')}")
    print(f"LoRA Dropout: {lora_config.get('lora_dropout', 0.1)}")
    
    # Training info
    training_config = config.get("training", {})
    print(f"Epochs: {training_config.get('num_train_epochs', 3)}")
    print(f"Batch Size: {training_config.get('per_device_train_batch_size', 4)}")
    print(f"Gradient Accumulation: {training_config.get('gradient_accumulation_steps', 4)}")
    print(f"Learning Rate: {training_config.get('learning_rate', 3e-4)}")
    print(f"LR Scheduler: {training_config.get('lr_scheduler_type', 'cosine')}")
    print(f"Mixed Precision: BF16={training_config.get('bf16', True)}, FP16={training_config.get('fp16', False)}")
    print(f"Gradient Checkpointing: {training_config.get('gradient_checkpointing', True)}")
    
    # Data info
    data_config = config.get("data", {})
    print(f"Max Sequence Length: {data_config.get('max_seq_length', 512)}")
    print(f"Train on Inputs: {data_config.get('train_on_inputs', False)}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    sample_config = {
        "model": {"model_name": "microsoft/DialoGPT-medium"},
        "lora": {"r": 16, "lora_alpha": 32},
        "training": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "learning_rate": 3e-4,
        },
        "data": {"max_seq_length": 512},
    }
    
    print_training_summary(sample_config) 