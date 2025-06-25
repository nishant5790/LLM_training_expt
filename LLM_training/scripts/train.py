#!/usr/bin/env python3
"""
Main training script for LoRA fine-tuning.

This script orchestrates the complete training pipeline including:
- Loading and preprocessing data
- Setting up model and LoRA adapters
- Training with monitoring and evaluation
- Saving the final model

Usage:
    python scripts/train.py --config config/training_config.yaml
    python scripts/train.py --config config/training_config.yaml --resume_from_checkpoint ./outputs/checkpoint-1000
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_preprocessing import DataProcessor, ChatDataCollator, create_sample_data
from model_setup import ModelSetup, print_model_info
from trainer import LoRATrainer, print_training_summary

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def setup_logging(config: dict):
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log")
        ]
    )


def prepare_output_directory(config: dict):
    """Create and prepare output directory."""
    output_dir = config.get("training", {}).get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config to output directory
    config_save_path = os.path.join(output_dir, "training_config.yaml")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Output directory prepared: {output_dir}")
    logger.info(f"Configuration saved to: {config_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train LoRA model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--create_sample_data",
        action="store_true",
        help="Create sample data if it doesn't exist"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with smaller dataset"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Print training summary
    print_training_summary(config)
    
    # Prepare output directory
    prepare_output_directory(config)
    
    try:
        logger.info("Starting LoRA training pipeline...")
        
        # Check if sample data needs to be created
        data_config = config.get("data", {})
        data_path = data_config.get("data_path", "./data/sample_data.json")
        
        if args.create_sample_data or not os.path.exists(data_path):
            logger.info("Creating sample data...")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            create_sample_data(data_path, 1000 if not args.debug else 50)
        
        # Setup model and tokenizer
        logger.info("Setting up model and tokenizer...")
        model_setup = ModelSetup(config)
        model, tokenizer = model_setup.setup_model_and_tokenizer()
        
        # Print model information
        print_model_info(model)
        
        # Setup data processor
        logger.info("Setting up data processor...")
        data_processor = DataProcessor(
            tokenizer=tokenizer,
            max_length=data_config.get("max_seq_length", 512),
            train_on_inputs=data_config.get("train_on_inputs", False),
        )
        
        # Prepare datasets
        logger.info("Preparing datasets...")
        datasets = data_processor.prepare_datasets(
            dataset_name=data_config.get("dataset_name", "local"),
            data_path=data_path,
            validation_split=data_config.get("validation_split", 0.1),
            num_proc=4,
        )
        
        # Debug mode: use smaller dataset
        if args.debug:
            logger.info("Debug mode: using smaller dataset")
            datasets["train"] = datasets["train"].select(range(min(100, len(datasets["train"]))))
            if "validation" in datasets:
                datasets["validation"] = datasets["validation"].select(range(min(50, len(datasets["validation"]))))
        
        logger.info(f"Training examples: {len(datasets['train'])}")
        if "validation" in datasets:
            logger.info(f"Validation examples: {len(datasets['validation'])}")
        
        # Setup data collator
        data_collator = ChatDataCollator(
            tokenizer=tokenizer,
            max_length=data_config.get("max_seq_length", 512),
            train_on_inputs=data_config.get("train_on_inputs", False),
        )
        
        # Setup trainer
        logger.info("Setting up trainer...")
        trainer = LoRATrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        
        # Start training
        logger.info("Starting training...")
        training_results = trainer.train(
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation"),
            data_collator=data_collator,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final results: {training_results}")
        
        # Save final model in different formats
        output_dir = config.get("training", {}).get("output_dir", "./outputs")
        
        # Save as PEFT adapter
        final_model_path = os.path.join(output_dir, "final_model")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        
        # Merge and save full model (optional)
        try:
            logger.info("Saving merged model...")
            merged_model = model.merge_and_unload()
            merged_model_path = os.path.join(output_dir, "merged_model")
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            logger.info(f"Merged model saved to: {merged_model_path}")
        except Exception as e:
            logger.warning(f"Failed to save merged model: {e}")
        
        logger.info("All training tasks completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 