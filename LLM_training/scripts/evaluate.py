#!/usr/bin/env python3
"""
Evaluation script for trained LoRA models.

This script evaluates a trained LoRA model on various metrics and test datasets.

Usage:
    python scripts/evaluate.py --model_path ./outputs/final_model
    python scripts/evaluate.py --model_path ./outputs/final_model --test_data ./data/test_data.json
"""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_preprocessing import DataProcessor, ChatDataCollator
from model_setup import ModelSetup
from trainer import LoRATrainer
from inference import LoRAInference, load_model_for_inference

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def load_test_data(test_data_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSON file."""
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} test examples from {test_data_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise


def create_test_prompts() -> List[Dict[str, str]]:
    """Create a set of test prompts for evaluation."""
    return [
        {
            "prompt": "Explain the concept of machine learning in simple terms.",
            "category": "technical_explanation"
        },
        {
            "prompt": "What are the benefits of renewable energy?",
            "category": "knowledge_question"
        },
        {
            "prompt": "Write a short story about a robot learning to paint.",
            "category": "creative_writing"
        },
        {
            "prompt": "How does photosynthesis work?",
            "category": "science_explanation"
        },
        {
            "prompt": "Describe the process of making coffee step by step.",
            "category": "process_description"
        },
        {
            "prompt": "What is the difference between AI and machine learning?",
            "category": "comparison"
        },
        {
            "prompt": "Give me three tips for effective time management.",
            "category": "advice"
        },
        {
            "prompt": "Explain quantum computing to a 10-year-old.",
            "category": "simplified_explanation"
        },
        {
            "prompt": "What are the main causes of climate change?",
            "category": "factual_question"
        },
        {
            "prompt": "Write a recipe for chocolate chip cookies.",
            "category": "instructions"
        },
    ]


def evaluate_response_quality(
    responses: List[str],
    prompts: List[str],
) -> Dict[str, float]:
    """
    Evaluate response quality using various metrics.
    
    Args:
        responses: Generated responses
        prompts: Original prompts
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Basic metrics
    response_lengths = [len(response.split()) for response in responses]
    metrics["avg_response_length"] = sum(response_lengths) / len(response_lengths)
    metrics["min_response_length"] = min(response_lengths)
    metrics["max_response_length"] = max(response_lengths)
    
    # Response quality indicators
    non_empty_responses = sum(1 for r in responses if r.strip())
    metrics["response_rate"] = non_empty_responses / len(responses)
    
    # Check for repetitive responses
    unique_responses = len(set(responses))
    metrics["response_diversity"] = unique_responses / len(responses)
    
    # Check for responses that are too short (likely incomplete)
    too_short = sum(1 for r in responses if len(r.split()) < 5)
    metrics["incomplete_rate"] = too_short / len(responses)
    
    # Check for responses that start with prompt (repetition issue)
    repetitive = sum(1 for i, r in enumerate(responses) 
                    if prompts[i].lower() in r.lower())
    metrics["repetition_rate"] = repetitive / len(responses)
    
    return metrics


def evaluate_perplexity(
    model_path: str,
    test_data: List[Dict[str, Any]],
    config_path: str = None,
) -> float:
    """
    Evaluate model perplexity on test data.
    
    Args:
        model_path: Path to trained model
        test_data: Test dataset
        config_path: Optional config file path
        
    Returns:
        Average perplexity
    """
    try:
        # Load config if provided
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        
        # Load model for evaluation
        model, tokenizer = ModelSetup.load_trained_model(model_path)
        
        # Setup data processor
        data_processor = DataProcessor(
            tokenizer=tokenizer,
            max_length=512,
            train_on_inputs=False,
        )
        
        # Convert test data to dataset format
        from datasets import Dataset
        dataset = Dataset.from_list(test_data)
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            data_processor.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        # Setup data collator
        data_collator = ChatDataCollator(
            tokenizer=tokenizer,
            max_length=512,
        )
        
        # Setup trainer for evaluation
        trainer = LoRATrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        
        # Evaluate
        eval_results = trainer.evaluate(
            eval_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        return eval_results.get("eval_perplexity", float("inf"))
        
    except Exception as e:
        logger.error(f"Failed to evaluate perplexity: {e}")
        return float("inf")


def run_comprehensive_evaluation(
    model_path: str,
    test_data_path: str = None,
    output_path: str = None,
    config_path: str = None,
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation of the trained model.
    
    Args:
        model_path: Path to trained model
        test_data_path: Optional path to test data
        output_path: Optional path to save results
        config_path: Optional config file path
        
    Returns:
        Evaluation results
    """
    logger.info("Starting comprehensive evaluation...")
    
    # Load model for inference
    inference = load_model_for_inference(model_path)
    
    # Prepare test data
    if test_data_path and os.path.exists(test_data_path):
        test_data = load_test_data(test_data_path)
        test_prompts = [item.get("prompt", item.get("instruction", "")) for item in test_data]
        categories = [item.get("category", "unknown") for item in test_data]
    else:
        logger.info("Using default test prompts")
        test_prompt_data = create_test_prompts()
        test_prompts = [item["prompt"] for item in test_prompt_data]
        categories = [item["category"] for item in test_prompt_data]
        test_data = test_prompt_data
    
    logger.info(f"Evaluating on {len(test_prompts)} prompts")
    
    # Generate responses
    logger.info("Generating responses...")
    responses = []
    response_times = []
    
    import time
    for prompt in test_prompts:
        start_time = time.time()
        response = inference.chat(prompt)
        end_time = time.time()
        
        responses.append(response)
        response_times.append(end_time - start_time)
    
    # Evaluate response quality
    logger.info("Evaluating response quality...")
    quality_metrics = evaluate_response_quality(responses, test_prompts)
    
    # Calculate timing metrics
    timing_metrics = {
        "avg_response_time": sum(response_times) / len(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "total_time": sum(response_times),
    }
    
    # Evaluate perplexity (if test data available)
    perplexity = None
    if test_data_path and os.path.exists(test_data_path):
        logger.info("Evaluating perplexity...")
        perplexity = evaluate_perplexity(model_path, test_data, config_path)
    
    # Compile results
    results = {
        "model_path": model_path,
        "num_test_prompts": len(test_prompts),
        "quality_metrics": quality_metrics,
        "timing_metrics": timing_metrics,
        "perplexity": perplexity,
        "detailed_results": [
            {
                "prompt": prompt,
                "response": response,
                "category": category,
                "response_time": time_taken,
                "response_length": len(response.split()),
            }
            for prompt, response, category, time_taken in 
            zip(test_prompts, responses, categories, response_times)
        ]
    }
    
    # Save results if output path provided
    if output_path:
        logger.info(f"Saving results to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Also save as CSV for easy analysis
        csv_path = output_path.replace('.json', '.csv')
        df = pd.DataFrame(results["detailed_results"])
        df.to_csv(csv_path, index=False)
        logger.info(f"Detailed results saved to {csv_path}")
    
    return results


def print_evaluation_summary(results: Dict[str, Any]):
    """Print a summary of evaluation results."""
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {results['model_path']}")
    print(f"Test Prompts: {results['num_test_prompts']}")
    
    # Quality metrics
    quality = results["quality_metrics"]
    print(f"\nResponse Quality:")
    print(f"  Average Length: {quality['avg_response_length']:.1f} words")
    print(f"  Response Rate: {quality['response_rate']:.2%}")
    print(f"  Diversity: {quality['response_diversity']:.2%}")
    print(f"  Incomplete Rate: {quality['incomplete_rate']:.2%}")
    print(f"  Repetition Rate: {quality['repetition_rate']:.2%}")
    
    # Timing metrics
    timing = results["timing_metrics"]
    print(f"\nTiming Performance:")
    print(f"  Average Response Time: {timing['avg_response_time']:.2f}s")
    print(f"  Total Time: {timing['total_time']:.2f}s")
    
    # Perplexity
    if results["perplexity"] is not None:
        print(f"\nPerplexity: {results['perplexity']:.2f}")
    
    # Category breakdown
    detailed = results["detailed_results"]
    categories = {}
    for item in detailed:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item["response_length"])
    
    print(f"\nCategory Breakdown:")
    for category, lengths in categories.items():
        avg_length = sum(lengths) / len(lengths)
        print(f"  {category}: {len(lengths)} prompts, avg {avg_length:.1f} words")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LoRA model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive evaluation mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        if args.interactive:
            # Interactive evaluation mode
            logger.info("Running interactive evaluation...")
            inference = load_model_for_inference(args.model_path)
            inference.interactive_chat()
        else:
            # Automated evaluation
            output_path = args.output or f"{args.model_path}/evaluation_results.json"
            
            results = run_comprehensive_evaluation(
                model_path=args.model_path,
                test_data_path=args.test_data,
                output_path=output_path,
                config_path=args.config,
            )
            
            # Print summary
            print_evaluation_summary(results)
            
            logger.info("Evaluation completed successfully!")
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main() 