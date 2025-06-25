#!/usr/bin/env python3
"""
Interactive inference demo for trained LoRA models.

This script provides various interactive modes for testing trained models:
- Chat mode: Interactive conversation
- Batch mode: Process multiple prompts
- Comparison mode: Compare different models
- Benchmark mode: Performance testing

Usage:
    python scripts/inference_demo.py --model_path ./outputs/final_model
    python scripts/inference_demo.py --model_path ./outputs/final_model --mode chat
    python scripts/inference_demo.py --model_path ./outputs/final_model --mode batch --prompts_file ./data/test_prompts.txt
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from inference import LoRAInference, load_model_for_inference

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def chat_mode(inference: LoRAInference):
    """Interactive chat mode."""
    print("\n" + "="*60)
    print("🤖 LoRA MODEL CHAT INTERFACE")
    print("="*60)
    print("Commands:")
    print("  'quit' or 'exit' - Exit the chat")
    print("  'clear' - Clear conversation history")
    print("  'help' - Show this help message")
    print("  'settings' - Show current generation settings")
    print("  'save <filename>' - Save conversation to file")
    print("="*60)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                conversation_history = []
                print("🧹 Conversation history cleared.")
                continue
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  quit/exit - Exit chat")
                print("  clear - Clear history")
                print("  settings - Show generation settings")
                print("  save <filename> - Save conversation")
                continue
            elif user_input.lower() == 'settings':
                config = inference.generation_config
                print(f"\nCurrent settings:")
                print(f"  Max tokens: {config.max_new_tokens}")
                print(f"  Temperature: {config.temperature}")
                print(f"  Top-p: {config.top_p}")
                print(f"  Top-k: {config.top_k}")
                continue
            elif user_input.lower().startswith('save '):
                filename = user_input[5:].strip()
                if filename and conversation_history:
                    inference.save_conversation(conversation_history, filename)
                    print(f"💾 Conversation saved to {filename}")
                else:
                    print("❌ Please provide a filename and ensure there's conversation to save")
                continue
            elif not user_input:
                continue
            
            # Generate response
            print("🤖 Assistant: ", end="", flush=True)
            start_time = time.time()
            
            # Add context from recent conversation
            if conversation_history:
                context = "\n\n".join(conversation_history[-6:])  # Last 6 exchanges
                full_prompt = f"{context}\n\nHuman: {user_input}\n\nAssistant:"
                response = inference.generate_response(full_prompt)
            else:
                response = inference.chat(user_input)
            
            end_time = time.time()
            
            print(response)
            print(f"⏱️  Response time: {end_time - start_time:.2f}s")
            
            # Add to history
            conversation_history.append(f"Human: {user_input}")
            conversation_history.append(f"Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def batch_mode(inference: LoRAInference, prompts_file: str, output_file: str = None):
    """Batch processing mode."""
    print("\n" + "="*60)
    print("📊 BATCH PROCESSING MODE")
    print("="*60)
    
    # Load prompts
    try:
        if prompts_file.endswith('.json'):
            with open(prompts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                if isinstance(data[0], dict):
                    prompts = [item.get('prompt', item.get('instruction', str(item))) for item in data]
                else:
                    prompts = data
            else:
                prompts = [data]
        else:
            # Text file with one prompt per line
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Error loading prompts: {e}")
        return
    
    print(f"📝 Processing {len(prompts)} prompts...")
    
    results = []
    total_time = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Processing: {prompt[:50]}...")
        
        start_time = time.time()
        response = inference.chat(prompt)
        end_time = time.time()
        
        response_time = end_time - start_time
        total_time += response_time
        
        result = {
            "prompt": prompt,
            "response": response,
            "response_time": response_time,
            "response_length": len(response.split())
        }
        results.append(result)
        
        print(f"✅ Completed in {response_time:.2f}s")
    
    # Print summary
    print(f"\n📊 BATCH SUMMARY:")
    print(f"Total prompts: {len(prompts)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per prompt: {total_time/len(prompts):.2f}s")
    print(f"Average response length: {sum(r['response_length'] for r in results)/len(results):.1f} words")
    
    # Save results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to {output_file}")
    
    return results


def comparison_mode(model_paths: List[str], test_prompts: List[str]):
    """Compare multiple models."""
    print("\n" + "="*60)
    print("🔄 MODEL COMPARISON MODE")
    print("="*60)
    
    # Load all models
    models = {}
    for path in model_paths:
        try:
            print(f"Loading model from {path}...")
            models[path] = load_model_for_inference(path)
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model from {path}: {e}")
    
    if not models:
        print("❌ No models loaded successfully")
        return
    
    print(f"\n🧪 Testing {len(models)} models on {len(test_prompts)} prompts...")
    
    results = {}
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        print("-" * 40)
        
        results[prompt] = {}
        
        for model_path, inference in models.items():
            start_time = time.time()
            response = inference.chat(prompt)
            end_time = time.time()
            
            results[prompt][model_path] = {
                "response": response,
                "time": end_time - start_time,
                "length": len(response.split())
            }
            
            print(f"\n🤖 {model_path}:")
            print(f"Response: {response}")
            print(f"Time: {end_time - start_time:.2f}s, Length: {len(response.split())} words")
    
    return results


def benchmark_mode(inference: LoRAInference, num_prompts: int = 10, prompt_length: int = 20):
    """Performance benchmark mode."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE BENCHMARK MODE")
    print("="*60)
    
    # Generate test prompts of varying lengths
    test_prompts = [
        "Explain artificial intelligence.",
        "What is machine learning and how does it work in modern applications?",
        "Write a detailed explanation of quantum computing, its principles, applications, and future potential in technology.",
        "Tell me about renewable energy sources and their impact on environmental sustainability, including solar, wind, and hydroelectric power generation methods.",
        "Provide a comprehensive overview of blockchain technology, including its underlying cryptographic principles, consensus mechanisms, real-world applications beyond cryptocurrency, and potential future developments in various industries."
    ]
    
    # Use provided prompts or generate more
    if num_prompts > len(test_prompts):
        base_prompt = "Explain the concept of"
        topics = ["science", "technology", "history", "literature", "mathematics", 
                 "physics", "chemistry", "biology", "psychology", "philosophy"]
        for i in range(num_prompts - len(test_prompts)):
            test_prompts.append(f"{base_prompt} {topics[i % len(topics)]} in detail.")
    
    test_prompts = test_prompts[:num_prompts]
    
    print(f"🧪 Running benchmark with {len(test_prompts)} prompts...")
    
    response_times = []
    response_lengths = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[{i}/{len(test_prompts)}] Processing prompt {i}...", end=" ")
        
        start_time = time.time()
        response = inference.chat(prompt)
        end_time = time.time()
        
        response_time = end_time - start_time
        response_length = len(response.split())
        
        response_times.append(response_time)
        response_lengths.append(response_length)
        
        print(f"Done ({response_time:.2f}s, {response_length} words)")
    
    # Calculate statistics
    avg_time = sum(response_times) / len(response_times)
    min_time = min(response_times)
    max_time = max(response_times)
    
    avg_length = sum(response_lengths) / len(response_lengths)
    min_length = min(response_lengths)
    max_length = max(response_lengths)
    
    total_time = sum(response_times)
    throughput = len(test_prompts) / total_time
    
    print(f"\n📊 BENCHMARK RESULTS:")
    print(f"Total prompts: {len(test_prompts)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} prompts/second")
    print(f"\nResponse Times:")
    print(f"  Average: {avg_time:.2f}s")
    print(f"  Min: {min_time:.2f}s")
    print(f"  Max: {max_time:.2f}s")
    print(f"\nResponse Lengths:")
    print(f"  Average: {avg_length:.1f} words")
    print(f"  Min: {min_length} words")
    print(f"  Max: {max_length} words")
    
    return {
        "total_prompts": len(test_prompts),
        "total_time": total_time,
        "throughput": throughput,
        "avg_response_time": avg_time,
        "min_response_time": min_time,
        "max_response_time": max_time,
        "avg_response_length": avg_length,
        "min_response_length": min_length,
        "max_response_length": max_length,
    }


def demo_mode(inference: LoRAInference):
    """Demonstration mode with predefined prompts."""
    print("\n" + "="*60)
    print("🎯 DEMONSTRATION MODE")
    print("="*60)
    
    demo_prompts = [
        {
            "prompt": "Explain machine learning to a 10-year-old.",
            "category": "Educational"
        },
        {
            "prompt": "Write a short poem about artificial intelligence.",
            "category": "Creative"
        },
        {
            "prompt": "What are the pros and cons of renewable energy?",
            "category": "Analysis"
        },
        {
            "prompt": "How do I make a perfect cup of coffee?",
            "category": "Instructions"
        },
        {
            "prompt": "Tell me a fascinating fact about space.",
            "category": "Knowledge"
        }
    ]
    
    print(f"🎬 Running demonstration with {len(demo_prompts)} prompts...\n")
    
    for i, item in enumerate(demo_prompts, 1):
        prompt = item["prompt"]
        category = item["category"]
        
        print(f"[{i}/{len(demo_prompts)}] {category} Example:")
        print(f"📝 Prompt: {prompt}")
        
        start_time = time.time()
        response = inference.chat(prompt)
        end_time = time.time()
        
        print(f"🤖 Response: {response}")
        print(f"⏱️  Time: {end_time - start_time:.2f}s")
        print("-" * 60)
        
        # Pause between demonstrations
        if i < len(demo_prompts):
            input("\nPress Enter to continue to next demonstration...")


def main():
    parser = argparse.ArgumentParser(description="Interactive inference demo for LoRA models")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["chat", "batch", "comparison", "benchmark", "demo"],
        default="chat",
        help="Demo mode to run"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        help="File containing prompts for batch mode"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for results"
    )
    parser.add_argument(
        "--compare_models",
        type=str,
        nargs="+",
        help="Multiple model paths for comparison mode"
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=10,
        help="Number of prompts for benchmark mode"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    try:
        if args.mode == "comparison" and args.compare_models:
            # Comparison mode
            test_prompts = [
                "Explain artificial intelligence.",
                "What are the benefits of renewable energy?",
                "Write a short story about a robot."
            ]
            comparison_mode(args.compare_models, test_prompts)
        else:
            # Load single model
            print(f"🔄 Loading model from {args.model_path}...")
            inference = load_model_for_inference(args.model_path)
            print("✅ Model loaded successfully!")
            
            if args.mode == "chat":
                chat_mode(inference)
            elif args.mode == "batch":
                if not args.prompts_file:
                    print("❌ Batch mode requires --prompts_file argument")
                    return
                batch_mode(inference, args.prompts_file, args.output_file)
            elif args.mode == "benchmark":
                benchmark_mode(inference, args.num_prompts)
            elif args.mode == "demo":
                demo_mode(inference)
    
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main() 