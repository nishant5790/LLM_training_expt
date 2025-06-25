#!/usr/bin/env python3
"""
Comprehensive comparison script for PEFT methods vs Full Fine-tuning.

This script compares different parameter-efficient fine-tuning methods:
- LoRA (Low-Rank Adaptation)
- AdaLoRA (Adaptive LoRA)
- Full Fine-tuning
- Frozen model (baseline)

Metrics compared:
- Memory usage
- Training time
- Model performance
- Parameter efficiency
- Inference speed

Usage:
    python scripts/compare_methods.py --config config/training_config.yaml
    python scripts/compare_methods.py --config config/training_config.yaml --methods lora adalora full
"""

import argparse
import json
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import psutil
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_preprocessing import DataProcessor, ChatDataCollator
from model_setup import ModelSetup
from trainer import LoRATrainer
from inference import LoRAInference
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, AdaLoraConfig, TaskType

logger = logging.getLogger(__name__)


class MethodComparator:
    """
    Class to compare different fine-tuning methods.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "./comparison_results"):
        self.config = config
        self.output_dir = output_dir
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    
    def measure_memory_usage(self, model: torch.nn.Module) -> Dict[str, float]:
        """Measure memory usage of a model."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Move model to GPU and measure
            model = model.cuda()
            
            # Get current and peak memory
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                "current_memory_gb": current_memory,
                "peak_memory_gb": peak_memory
            }
        else:
            # For CPU, measure using psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024**2
            return {
                "current_memory_gb": memory_mb / 1024,
                "peak_memory_gb": memory_mb / 1024
            }
    
    def count_parameters(self, model: torch.nn.Module) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100
        }
    
    def setup_lora_model(self, base_model, tokenizer) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Setup LoRA model."""
        logger.info("Setting up LoRA model...")
        
        lora_config = LoraConfig(
            r=self.config["lora"]["r"],
            lora_alpha=self.config["lora"]["lora_alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            bias=self.config["lora"]["bias"],
            task_type=getattr(TaskType, self.config["lora"]["task_type"])
        )
        
        model = get_peft_model(base_model, lora_config)
        
        return model, {
            "method": "LoRA",
            "rank": self.config["lora"]["r"],
            "alpha": self.config["lora"]["lora_alpha"],
            "target_modules": self.config["lora"]["target_modules"]
        }
    
    def setup_adalora_model(self, base_model, tokenizer) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Setup AdaLoRA model."""
        logger.info("Setting up AdaLoRA model...")
        
        adalora_config = AdaLoraConfig(
            target_modules=self.config["lora"]["target_modules"],
            init_r=12,
            target_r=8,
            beta1=0.85,
            beta2=0.85,
            tinit=0,
            tfinal=1000,
            deltaT=10,
            lora_alpha=self.config["lora"]["lora_alpha"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            task_type=getattr(TaskType, self.config["lora"]["task_type"])
        )
        
        model = get_peft_model(base_model, adalora_config)
        
        return model, {
            "method": "AdaLoRA",
            "init_rank": 12,
            "target_rank": 8,
            "target_modules": self.config["lora"]["target_modules"]
        }
    
    def setup_full_finetune_model(self, base_model, tokenizer) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Setup full fine-tuning model."""
        logger.info("Setting up full fine-tuning model...")
        
        # Unfreeze all parameters
        for param in base_model.parameters():
            param.requires_grad = True
        
        return base_model, {
            "method": "Full Fine-tuning",
            "unfrozen_layers": "all"
        }
    
    def setup_frozen_model(self, base_model, tokenizer) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """Setup frozen model (baseline)."""
        logger.info("Setting up frozen model (baseline)...")
        
        # Freeze all parameters
        for param in base_model.parameters():
            param.requires_grad = False
        
        return base_model, {
            "method": "Frozen (Baseline)",
            "trainable_params": 0
        }
    
    def train_and_evaluate_method(
        self, 
        method_name: str, 
        train_dataset, 
        eval_dataset, 
        data_collator
    ) -> Dict[str, Any]:
        """Train and evaluate a specific method."""
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING METHOD: {method_name.upper()}")
        logger.info(f"{'='*60}")
        
        # Setup model based on method
        model_setup = ModelSetup(self.config)
        base_model, tokenizer = model_setup.load_base_model(model_setup.load_tokenizer()), model_setup.load_tokenizer()
        
        if method_name == "lora":
            model, method_info = self.setup_lora_model(base_model, tokenizer)
        elif method_name == "adalora":
            model, method_info = self.setup_adalora_model(base_model, tokenizer)
        elif method_name == "full":
            model, method_info = self.setup_full_finetune_model(base_model, tokenizer)
        elif method_name == "frozen":
            model, method_info = self.setup_frozen_model(base_model, tokenizer)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Measure initial memory and parameters
        param_info = self.count_parameters(model)
        memory_before = self.measure_memory_usage(model)
        
        logger.info(f"Parameters: {param_info['trainable_params']:,} trainable / {param_info['total_params']:,} total ({param_info['trainable_percentage']:.2f}%)")
        logger.info(f"Memory before training: {memory_before['current_memory_gb']:.2f} GB")
        
        # Training setup
        training_config = self.config["training"].copy()
        training_config["output_dir"] = os.path.join(self.output_dir, f"{method_name}_model")
        training_config["num_train_epochs"] = 1  # Reduced for comparison
        training_config["save_strategy"] = "no"  # Don't save checkpoints
        training_config["evaluation_strategy"] = "epoch"
        training_config["logging_steps"] = 10
        
        # Measure training time
        start_time = time.time()
        
        try:
            if method_name == "frozen":
                # For frozen model, skip training
                training_results = {"train_loss": float("inf"), "train_runtime": 0}
                eval_results = {"eval_loss": float("inf")}
            else:
                # Setup trainer
                if method_name in ["lora", "adalora"]:
                    trainer = LoRATrainer(model, tokenizer, {"training": training_config})
                    training_results = trainer.train(train_dataset, eval_dataset, data_collator)
                    eval_results = trainer.evaluate(eval_dataset, data_collator)
                else:
                    # Full fine-tuning with standard trainer
                    training_args = TrainingArguments(**training_config)
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=data_collator,
                        tokenizer=tokenizer
                    )
                    train_result = trainer.train()
                    training_results = train_result.metrics
                    eval_results = trainer.evaluate()
            
            training_time = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Training failed for {method_name}: {e}")
            training_results = {"train_loss": float("inf"), "train_runtime": 0}
            eval_results = {"eval_loss": float("inf")}
            training_time = 0
        
        # Measure memory after training
        memory_after = self.measure_memory_usage(model)
        
        # Measure inference speed
        inference_times = self.measure_inference_speed(model, tokenizer)
        
        # Compile results
        results = {
            **method_info,
            **param_info,
            "memory_before_gb": memory_before["current_memory_gb"],
            "memory_after_gb": memory_after["current_memory_gb"],
            "peak_memory_gb": memory_after["peak_memory_gb"],
            "training_time_seconds": training_time,
            "train_loss": training_results.get("train_loss", float("inf")),
            "eval_loss": eval_results.get("eval_loss", float("inf")),
            "avg_inference_time": inference_times["avg_time"],
            "inference_throughput": inference_times["throughput"]
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final train loss: {results['train_loss']:.4f}")
        logger.info(f"Final eval loss: {results['eval_loss']:.4f}")
        logger.info(f"Peak memory usage: {results['peak_memory_gb']:.2f} GB")
        
        return results
    
    def measure_inference_speed(self, model, tokenizer, num_samples: int = 10) -> Dict[str, float]:
        """Measure inference speed."""
        model.eval()
        
        test_prompts = [
            "Explain machine learning.",
            "What is artificial intelligence?",
            "How does deep learning work?",
            "What are neural networks?",
            "Describe natural language processing."
        ]
        
        times = []
        
        with torch.no_grad():
            for i in range(num_samples):
                prompt = test_prompts[i % len(test_prompts)]
                formatted_prompt = f"Human: {prompt}\n\nAssistant:"
                
                inputs = tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                start_time = time.time()
                
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    end_time = time.time()
                    times.append(end_time - start_time)
                except Exception as e:
                    logger.warning(f"Inference failed: {e}")
                    times.append(float("inf"))
        
        valid_times = [t for t in times if t != float("inf")]
        if valid_times:
            avg_time = sum(valid_times) / len(valid_times)
            throughput = 1.0 / avg_time if avg_time > 0 else 0
        else:
            avg_time = float("inf")
            throughput = 0
        
        return {
            "avg_time": avg_time,
            "throughput": throughput
        }
    
    def compare_methods(self, methods: List[str]) -> Dict[str, Any]:
        """Compare multiple methods."""
        logger.info("Starting method comparison...")
        
        # Prepare datasets
        model_setup = ModelSetup(self.config)
        tokenizer = model_setup.load_tokenizer()
        
        data_processor = DataProcessor(
            tokenizer=tokenizer,
            max_length=self.config["data"]["max_seq_length"],
            train_on_inputs=self.config["data"]["train_on_inputs"]
        )
        
        datasets = data_processor.prepare_datasets(
            dataset_name=self.config["data"]["dataset_name"],
            data_path=self.config["data"]["data_path"],
            validation_split=self.config["data"]["validation_split"]
        )
        
        # Use smaller datasets for comparison
        train_dataset = datasets["train"].select(range(min(100, len(datasets["train"]))))
        eval_dataset = datasets["validation"].select(range(min(50, len(datasets["validation"]))))
        
        data_collator = ChatDataCollator(
            tokenizer=tokenizer,
            max_length=self.config["data"]["max_seq_length"]
        )
        
        # Run comparison for each method
        results = {}
        for method in methods:
            try:
                results[method] = self.train_and_evaluate_method(
                    method, train_dataset, eval_dataset, data_collator
                )
            except Exception as e:
                logger.error(f"Failed to evaluate {method}: {e}")
                results[method] = {
                    "method": method,
                    "error": str(e),
                    "train_loss": float("inf"),
                    "eval_loss": float("inf")
                }
        
        return results
    
    def create_comparison_report(self, results: Dict[str, Any]):
        """Create detailed comparison report with visualizations."""
        logger.info("Creating comparison report...")
        
        # Convert results to DataFrame
        df_data = []
        for method, result in results.items():
            if "error" not in result:
                df_data.append({
                    "Method": result.get("method", method),
                    "Trainable Params (M)": result.get("trainable_params", 0) / 1e6,
                    "Trainable %": result.get("trainable_percentage", 0),
                    "Memory (GB)": result.get("peak_memory_gb", 0),
                    "Training Time (s)": result.get("training_time_seconds", 0),
                    "Train Loss": result.get("train_loss", float("inf")),
                    "Eval Loss": result.get("eval_loss", float("inf")),
                    "Inference Time (s)": result.get("avg_inference_time", 0),
                    "Throughput (req/s)": result.get("inference_throughput", 0)
                })
        
        df = pd.DataFrame(df_data)
        
        # Save detailed results
        with open(os.path.join(self.output_dir, "detailed_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        df.to_csv(os.path.join(self.output_dir, "comparison_summary.csv"), index=False)
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Generate text report
        self.generate_text_report(df, results)
        
        logger.info(f"Comparison report saved to {self.output_dir}")
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create comparison visualizations."""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PEFT Methods Comparison', fontsize=16, fontweight='bold')
        
        # 1. Parameter Efficiency
        ax1 = axes[0, 0]
        bars1 = ax1.bar(df['Method'], df['Trainable Params (M)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Trainable Parameters (Millions)', fontweight='bold')
        ax1.set_ylabel('Parameters (M)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}M', ha='center', va='bottom')
        
        # 2. Memory Usage
        ax2 = axes[0, 1]
        bars2 = ax2.bar(df['Method'], df['Memory (GB)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Peak Memory Usage', fontweight='bold')
        ax2.set_ylabel('Memory (GB)')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}GB', ha='center', va='bottom')
        
        # 3. Training Time
        ax3 = axes[0, 2]
        bars3 = ax3.bar(df['Method'], df['Training Time (s)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax3.set_title('Training Time', fontweight='bold')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom')
        
        # 4. Training Loss
        ax4 = axes[1, 0]
        valid_losses = df[df['Train Loss'] != float('inf')]
        if not valid_losses.empty:
            bars4 = ax4.bar(valid_losses['Method'], valid_losses['Train Loss'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax4.set_title('Final Training Loss', fontweight='bold')
            ax4.set_ylabel('Loss')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # 5. Inference Speed
        ax5 = axes[1, 1]
        valid_inference = df[df['Inference Time (s)'] != float('inf')]
        if not valid_inference.empty:
            bars5 = ax5.bar(valid_inference['Method'], valid_inference['Inference Time (s)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax5.set_title('Inference Time per Request', fontweight='bold')
            ax5.set_ylabel('Time (seconds)')
            ax5.tick_params(axis='x', rotation=45)
            
            for bar in bars5:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}s', ha='center', va='bottom')
        
        # 6. Efficiency Score (composite metric)
        ax6 = axes[1, 2]
        # Calculate efficiency score: inverse of (memory * time * parameters)
        df['Efficiency Score'] = 1 / (df['Memory (GB)'] * df['Training Time (s)'] * (df['Trainable Params (M)'] + 1))
        df['Efficiency Score'] = df['Efficiency Score'] / df['Efficiency Score'].max() * 100  # Normalize to 0-100
        
        bars6 = ax6.bar(df['Method'], df['Efficiency Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax6.set_title('Overall Efficiency Score', fontweight='bold')
        ax6.set_ylabel('Score (0-100)')
        ax6.tick_params(axis='x', rotation=45)
        
        for bar in bars6:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "method_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a radar chart for multi-dimensional comparison
        self.create_radar_chart(df)
    
    def create_radar_chart(self, df: pd.DataFrame):
        """Create radar chart for multi-dimensional comparison."""
        from math import pi
        
        # Prepare data for radar chart (normalize metrics)
        metrics = ['Trainable %', 'Memory (GB)', 'Training Time (s)', 'Inference Time (s)']
        
        # Normalize metrics (invert some so higher is better)
        df_radar = df.copy()
        df_radar['Memory (GB)'] = 1 / (df_radar['Memory (GB)'] + 0.1)  # Invert (lower is better)
        df_radar['Training Time (s)'] = 1 / (df_radar['Training Time (s)'] + 0.1)  # Invert
        df_radar['Inference Time (s)'] = 1 / (df_radar['Inference Time (s)'] + 0.1)  # Invert
        df_radar['Trainable %'] = 1 / (df_radar['Trainable %'] + 0.1)  # Invert (fewer params is better)
        
        # Normalize to 0-1 scale
        for metric in metrics:
            if metric in df_radar.columns:
                max_val = df_radar[metric].max()
                if max_val > 0:
                    df_radar[metric] = df_radar[metric] / max_val
        
        # Setup radar chart
        angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (_, row) in enumerate(df_radar.iterrows()):
            if i < len(colors):
                values = [row[metric] for metric in metrics]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=row['Method'], color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Dimensional Method Comparison\n(Higher is Better)', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "radar_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_text_report(self, df: pd.DataFrame, results: Dict[str, Any]):
        """Generate detailed text report."""
        report_path = os.path.join(self.output_dir, "comparison_report.md")
        
        with open(report_path, "w") as f:
            f.write("# PEFT Methods Comparison Report\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report compares different Parameter-Efficient Fine-Tuning (PEFT) methods ")
            f.write("against full fine-tuning and a frozen baseline model.\n\n")
            
            f.write("## Methods Compared\n\n")
            for method, result in results.items():
                if "error" not in result:
                    f.write(f"- **{result.get('method', method)}**: ")
                    if method == "lora":
                        f.write(f"Rank={result.get('rank', 'N/A')}, Alpha={result.get('alpha', 'N/A')}\n")
                    elif method == "adalora":
                        f.write(f"Adaptive rank from {result.get('init_rank', 'N/A')} to {result.get('target_rank', 'N/A')}\n")
                    elif method == "full":
                        f.write("All parameters trainable\n")
                    elif method == "frozen":
                        f.write("No parameters trainable (baseline)\n")
                    else:
                        f.write("\n")
            
            f.write("\n## Detailed Results\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".3f"))
            f.write("\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Parameter efficiency
            most_efficient = df.loc[df['Trainable Params (M)'].idxmin()]
            f.write(f"- **Most Parameter Efficient**: {most_efficient['Method']} ")
            f.write(f"({most_efficient['Trainable Params (M)']:.2f}M parameters, ")
            f.write(f"{most_efficient['Trainable %']:.2f}% of total)\n")
            
            # Memory efficiency
            most_memory_efficient = df.loc[df['Memory (GB)'].idxmin()]
            f.write(f"- **Most Memory Efficient**: {most_memory_efficient['Method']} ")
            f.write(f"({most_memory_efficient['Memory (GB)']:.2f} GB peak usage)\n")
            
            # Training speed
            fastest_training = df.loc[df['Training Time (s)'].idxmin()]
            f.write(f"- **Fastest Training**: {fastest_training['Method']} ")
            f.write(f"({fastest_training['Training Time (s)']:.2f} seconds)\n")
            
            # Best performance
            valid_performance = df[df['Eval Loss'] != float('inf')]
            if not valid_performance.empty:
                best_performance = valid_performance.loc[valid_performance['Eval Loss'].idxmin()]
                f.write(f"- **Best Performance**: {best_performance['Method']} ")
                f.write(f"(Eval Loss: {best_performance['Eval Loss']:.4f})\n")
            
            f.write("\n## Advantages and Disadvantages\n\n")
            
            f.write("### LoRA (Low-Rank Adaptation)\n")
            f.write("**Advantages:**\n")
            f.write("- ✅ Extremely parameter efficient (typically <1% of original parameters)\n")
            f.write("- ✅ Fast training and low memory usage\n")
            f.write("- ✅ Easy to deploy and swap different adapters\n")
            f.write("- ✅ Original model weights preserved\n")
            f.write("- ✅ Good performance on most tasks\n\n")
            
            f.write("**Disadvantages:**\n")
            f.write("- ❌ May underperform on complex tasks requiring significant model changes\n")
            f.write("- ❌ Hyperparameter tuning required (rank, alpha)\n")
            f.write("- ❌ Limited to linear layer adaptations\n\n")
            
            f.write("### AdaLoRA (Adaptive LoRA)\n")
            f.write("**Advantages:**\n")
            f.write("- ✅ Automatically adjusts rank allocation\n")
            f.write("- ✅ Better parameter utilization than fixed LoRA\n")
            f.write("- ✅ Can achieve better performance with same parameter budget\n\n")
            
            f.write("**Disadvantages:**\n")
            f.write("- ❌ More complex setup and hyperparameters\n")
            f.write("- ❌ Slightly slower training than standard LoRA\n")
            f.write("- ❌ Less mature and tested than LoRA\n\n")
            
            f.write("### Full Fine-tuning\n")
            f.write("**Advantages:**\n")
            f.write("- ✅ Maximum adaptability to new tasks\n")
            f.write("- ✅ Can achieve best possible performance\n")
            f.write("- ✅ Simple setup (just unfreeze parameters)\n\n")
            
            f.write("**Disadvantages:**\n")
            f.write("- ❌ Requires massive memory and compute resources\n")
            f.write("- ❌ Slow training\n")
            f.write("- ❌ Risk of catastrophic forgetting\n")
            f.write("- ❌ Large checkpoint sizes\n")
            f.write("- ❌ Expensive to deploy multiple task-specific models\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("**Use LoRA when:**\n")
            f.write("- Working with limited computational resources\n")
            f.write("- Need to quickly adapt to multiple tasks\n")
            f.write("- Want to preserve original model capabilities\n")
            f.write("- Deploying multiple task-specific variants\n\n")
            
            f.write("**Use AdaLoRA when:**\n")
            f.write("- Have slightly more computational budget\n")
            f.write("- Need better parameter utilization\n")
            f.write("- Working on complex adaptation tasks\n\n")
            
            f.write("**Use Full Fine-tuning when:**\n")
            f.write("- Have abundant computational resources\n")
            f.write("- Need maximum possible performance\n")
            f.write("- Task requires significant model changes\n")
            f.write("- Working with very different domains from pre-training\n\n")
            
            f.write("## Technical Details\n\n")
            f.write(f"- **Base Model**: {self.config['model']['model_name']}\n")
            f.write(f"- **Training Examples**: {len(df)} methods compared\n")
            f.write(f"- **Evaluation Metric**: Cross-entropy loss\n")
            f.write(f"- **Hardware**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"- **Framework**: PyTorch + Transformers + PEFT\n\n")
            
        logger.info(f"Detailed report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare PEFT methods")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["lora", "adalora", "full", "frozen"],
        choices=["lora", "adalora", "full", "frozen"],
        help="Methods to compare"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./comparison_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run comparison
    comparator = MethodComparator(config, args.output_dir)
    results = comparator.compare_methods(args.methods)
    comparator.create_comparison_report(results)
    
    print(f"\n🎉 Comparison completed!")
    print(f"📊 Results saved to: {args.output_dir}")
    print(f"📈 Visualizations: {args.output_dir}/method_comparison.png")
    print(f"📋 Detailed report: {args.output_dir}/comparison_report.md")


if __name__ == "__main__":
    main() 