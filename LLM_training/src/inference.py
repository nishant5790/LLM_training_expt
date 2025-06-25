"""
Inference utilities for LoRA fine-tuned models.

This module provides utilities for loading trained LoRA models
and running inference for text generation and chat interactions.
"""

import logging
import torch
from typing import Dict, List, Optional, Union, Any
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
)
from peft import PeftModel
import json

logger = logging.getLogger(__name__)


class LoRAInference:
    """
    Inference class for LoRA fine-tuned models.
    """
    
    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        self.inference_config = self.config.get("inference", {})
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Configure generation parameters
        self.generation_config = self._create_generation_config()
        
        logger.info("LoRA inference model initialized")
    
    def _create_generation_config(self) -> GenerationConfig:
        """Create generation configuration from config."""
        return GenerationConfig(
            max_new_tokens=self.inference_config.get("max_new_tokens", 256),
            temperature=self.inference_config.get("temperature", 0.7),
            top_p=self.inference_config.get("top_p", 0.9),
            top_k=self.inference_config.get("top_k", 50),
            do_sample=self.inference_config.get("do_sample", True),
            repetition_penalty=self.inference_config.get("repetition_penalty", 1.1),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
    
    def format_chat_prompt(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Format a message into chat format.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        if system_prompt:
            return f"System: {system_prompt}\n\nHuman: {message}\n\nAssistant:"
        else:
            return f"Human: {message}\n\nAssistant:"
    
    def format_instruction_prompt(
        self, 
        instruction: str, 
        input_text: Optional[str] = None
    ) -> str:
        """
        Format an instruction-based prompt.
        
        Args:
            instruction: Instruction text
            input_text: Optional input text
            
        Returns:
            Formatted prompt string
        """
        if input_text:
            return f"Human: {instruction}\n\nInput: {input_text}\n\nAssistant:"
        else:
            return f"Human: {instruction}\n\nAssistant:"
    
    def generate_response(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        return_full_text: bool = False,
    ) -> str:
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: Input prompt
            generation_config: Optional custom generation config
            return_full_text: Whether to return full text or just the generated part
            
        Returns:
            Generated response
        """
        # Use custom generation config if provided
        gen_config = generation_config or self.generation_config
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # Reasonable input length
        )
        
        # Move to model device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode response
        if return_full_text:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Only return the generated part
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """
        Simple chat interface.
        
        Args:
            message: User message
            system_prompt: Optional system prompt
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Assistant response
        """
        prompt = self.format_chat_prompt(message, system_prompt)
        
        # Create custom generation config if kwargs provided
        if generation_kwargs:
            gen_config = GenerationConfig(**{**self.generation_config.to_dict(), **generation_kwargs})
        else:
            gen_config = self.generation_config
        
        return self.generate_response(prompt, gen_config)
    
    def instruct(
        self,
        instruction: str,
        input_text: Optional[str] = None,
        **generation_kwargs
    ) -> str:
        """
        Instruction-following interface.
        
        Args:
            instruction: Instruction text
            input_text: Optional input text
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        prompt = self.format_instruction_prompt(instruction, input_text)
        
        # Create custom generation config if kwargs provided
        if generation_kwargs:
            gen_config = GenerationConfig(**{**self.generation_config.to_dict(), **generation_kwargs})
        else:
            gen_config = self.generation_config
        
        return self.generate_response(prompt, gen_config)
    
    def batch_generate(
        self,
        prompts: List[str],
        generation_config: Optional[GenerationConfig] = None,
        batch_size: int = 4,
        return_full_text: bool = False,
    ) -> List[str]:
        """
        Generate responses for multiple prompts in batches.
        
        Args:
            prompts: List of input prompts
            generation_config: Optional custom generation config
            batch_size: Batch size for processing
            return_full_text: Whether to return full text
            
        Returns:
            List of generated responses
        """
        gen_config = generation_config or self.generation_config
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            
            # Move to device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode responses
            for j, output in enumerate(outputs):
                if return_full_text:
                    response = self.tokenizer.decode(output, skip_special_tokens=True)
                else:
                    input_length = inputs["input_ids"][j].shape[0]
                    generated_tokens = output[input_length:]
                    response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                responses.append(response.strip())
        
        return responses
    
    def interactive_chat(self):
        """
        Run an interactive chat session.
        """
        print("LoRA Chat Interface")
        print("Type 'quit' to exit, 'clear' to clear context")
        print("=" * 50)
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    print("Conversation history cleared.")
                    continue
                elif not user_input:
                    continue
                
                # Add user message to history
                conversation_history.append(f"Human: {user_input}")
                
                # Create prompt with context
                context = "\n\n".join(conversation_history[-6:])  # Keep last 6 turns
                prompt = f"{context}\n\nAssistant:"
                
                # Generate response
                response = self.generate_response(prompt)
                
                # Add assistant response to history
                conversation_history.append(f"Assistant: {response}")
                
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def evaluate_prompt_quality(
        self,
        prompts: List[str],
        expected_responses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of generated responses.
        
        Args:
            prompts: List of test prompts
            expected_responses: Optional expected responses for comparison
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {len(prompts)} prompts...")
        
        # Generate responses
        responses = self.batch_generate(prompts)
        
        # Basic metrics
        metrics = {
            "total_prompts": len(prompts),
            "avg_response_length": sum(len(r.split()) for r in responses) / len(responses),
            "responses": responses,
        }
        
        # If expected responses provided, compute similarity
        if expected_responses:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Compute TF-IDF similarity
            vectorizer = TfidfVectorizer()
            all_texts = responses + expected_responses
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            similarities = []
            for i in range(len(responses)):
                sim = cosine_similarity(
                    tfidf_matrix[i:i+1], 
                    tfidf_matrix[len(responses)+i:len(responses)+i+1]
                )[0][0]
                similarities.append(sim)
            
            metrics["avg_similarity"] = sum(similarities) / len(similarities)
            metrics["similarities"] = similarities
        
        return metrics
    
    def save_conversation(
        self,
        conversation: List[Dict[str, str]],
        output_path: str,
    ):
        """
        Save a conversation to file.
        
        Args:
            conversation: List of message dictionaries
            output_path: Path to save the conversation
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversation saved to {output_path}")
    
    def load_conversation(self, input_path: str) -> List[Dict[str, str]]:
        """
        Load a conversation from file.
        
        Args:
            input_path: Path to load the conversation from
            
        Returns:
            List of message dictionaries
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
        
        logger.info(f"Conversation loaded from {input_path}")
        return conversation


def load_model_for_inference(
    model_path: str,
    base_model_name: Optional[str] = None,
    device: str = "auto",
) -> LoRAInference:
    """
    Load a trained LoRA model for inference.
    
    Args:
        model_path: Path to the trained model
        base_model_name: Base model name if different from saved config
        device: Device to load model on
        
    Returns:
        LoRAInference instance
    """
    from .model_setup import ModelSetup
    
    # Load the trained model
    model, tokenizer = ModelSetup.load_trained_model(model_path, base_model_name)
    
    # Create inference instance
    inference = LoRAInference(model, tokenizer)
    
    return inference


def run_inference_demo(model_path: str, base_model_name: Optional[str] = None):
    """
    Run a simple inference demonstration.
    
    Args:
        model_path: Path to the trained model
        base_model_name: Base model name if different
    """
    print("Loading model for inference...")
    inference = load_model_for_inference(model_path, base_model_name)
    
    print("Model loaded successfully!")
    
    # Demo prompts
    demo_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "What are the benefits of renewable energy?",
        "How does photosynthesis work?",
        "Write a short story about a robot learning to paint.",
    ]
    
    print("\nDemo Responses:")
    print("=" * 50)
    
    for prompt in demo_prompts:
        print(f"\nPrompt: {prompt}")
        response = inference.chat(prompt)
        print(f"Response: {response}")
        print("-" * 30)
    
    # Interactive mode
    print("\nEntering interactive mode...")
    inference.interactive_chat()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        base_model = sys.argv[2] if len(sys.argv) > 2 else None
        run_inference_demo(model_path, base_model)
    else:
        print("Usage: python inference.py <model_path> [base_model_name]") 