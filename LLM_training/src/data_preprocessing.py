"""
Data preprocessing utilities for LoRA fine-tuning.

This module handles loading, preprocessing, and formatting of datasets
for instruction tuning and chat-based fine-tuning.
"""

import json
import logging
import random
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ChatDataCollator:
    """
    Data collator for chat/instruction datasets with proper attention masking.
    """
    tokenizer: PreTrainedTokenizer
    max_length: int = 512
    train_on_inputs: bool = False
    ignore_index: int = -100

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of examples with proper padding and attention masks.
        """
        # Extract input_ids and labels
        input_ids = [example["input_ids"] for example in batch]
        labels = [example["labels"] for example in batch]
        
        # Pad sequences
        input_ids = self._pad_sequences(input_ids, self.tokenizer.pad_token_id)
        labels = self._pad_sequences(labels, self.ignore_index)
        
        # Create attention masks
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _pad_sequences(self, sequences: List[List[int]], pad_value: int) -> torch.Tensor:
        """Pad sequences to the same length."""
        max_len = max(len(seq) for seq in sequences)
        max_len = min(max_len, self.max_length)  # Respect max_length
        
        padded = []
        for seq in sequences:
            seq = seq[:max_len]  # Truncate if needed
            padded_seq = seq + [pad_value] * (max_len - len(seq))
            padded.append(padded_seq)
        
        return torch.tensor(padded, dtype=torch.long)


class DataProcessor:
    """
    Main data processing class for instruction tuning datasets.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        train_on_inputs: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_on_inputs = train_on_inputs
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_dataset(
        self,
        dataset_name: str,
        data_path: Optional[str] = None,
        validation_split: float = 0.1,
    ) -> Dict[str, Dataset]:
        """
        Load dataset from various sources.
        
        Args:
            dataset_name: Name of the dataset or "local" for custom data
            data_path: Path to local data file
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with train and validation datasets
        """
        if dataset_name == "local" and data_path:
            return self._load_local_dataset(data_path, validation_split)
        else:
            return self._load_huggingface_dataset(dataset_name, validation_split)
    
    def _load_local_dataset(
        self, 
        data_path: str, 
        validation_split: float
    ) -> Dict[str, Dataset]:
        """Load dataset from local JSON file."""
        logger.info(f"Loading local dataset from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data)
        
        # Split into train/validation
        if validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=validation_split, seed=42)
            return {
                "train": split_dataset["train"],
                "validation": split_dataset["test"]
            }
        else:
            return {"train": dataset}
    
    def _load_huggingface_dataset(
        self, 
        dataset_name: str, 
        validation_split: float
    ) -> Dict[str, Dataset]:
        """Load dataset from Hugging Face hub."""
        logger.info(f"Loading dataset from Hugging Face: {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        # Handle different dataset structures
        if "train" in dataset:
            train_dataset = dataset["train"]
            if "validation" in dataset:
                val_dataset = dataset["validation"]
            elif validation_split > 0:
                split = train_dataset.train_test_split(test_size=validation_split, seed=42)
                train_dataset = split["train"]
                val_dataset = split["test"]
            else:
                val_dataset = None
        else:
            # Single split dataset
            if validation_split > 0:
                split = dataset.train_test_split(test_size=validation_split, seed=42)
                train_dataset = split["train"]
                val_dataset = split["test"]
            else:
                train_dataset = dataset
                val_dataset = None
        
        result = {"train": train_dataset}
        if val_dataset:
            result["validation"] = val_dataset
        
        return result
    
    def format_chat_example(self, example: Dict[str, Any]) -> str:
        """
        Format a single example into chat format.
        
        Expected input formats:
        1. {"instruction": "...", "input": "...", "output": "..."}
        2. {"prompt": "...", "response": "..."}
        3. {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        """
        if "messages" in example:
            return self._format_messages(example["messages"])
        elif "instruction" in example:
            return self._format_instruction(example)
        elif "prompt" in example and "response" in example:
            return f"Human: {example['prompt']}\n\nAssistant: {example['response']}"
        else:
            raise ValueError(f"Unknown example format: {example.keys()}")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages list into chat format."""
        formatted = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user" or role == "human":
                formatted += f"Human: {content}\n\n"
            elif role == "assistant" or role == "ai":
                formatted += f"Assistant: {content}"
            elif role == "system":
                formatted = f"System: {content}\n\n" + formatted
        
        return formatted
    
    def _format_instruction(self, example: Dict[str, Any]) -> str:
        """Format instruction-based example."""
        instruction = example["instruction"]
        input_text = example.get("input", "")
        output = example["output"]
        
        if input_text:
            prompt = f"Human: {instruction}\n\nInput: {input_text}\n\nAssistant: {output}"
        else:
            prompt = f"Human: {instruction}\n\nAssistant: {output}"
        
        return prompt
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Tokenize examples for training.
        
        This function processes the examples and creates input_ids and labels
        with proper masking for instruction tuning.
        """
        batch_input_ids = []
        batch_labels = []
        
        for i in range(len(examples[list(examples.keys())[0]])):
            # Get single example
            example = {key: examples[key][i] for key in examples.keys()}
            
            # Format the example
            text = self.format_chat_example(example)
            
            # Tokenize
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None,
            )
            
            input_ids = tokenized["input_ids"]
            
            # Create labels
            if self.train_on_inputs:
                # Train on entire sequence
                labels = input_ids.copy()
            else:
                # Only train on assistant responses
                labels = self._mask_input_tokens(text, input_ids)
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
        
        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
        }
    
    def _mask_input_tokens(self, text: str, input_ids: List[int]) -> List[int]:
        """
        Mask input tokens so the model only trains on assistant responses.
        """
        labels = input_ids.copy()
        
        # Find assistant response start
        assistant_start = text.find("Assistant:")
        if assistant_start == -1:
            # If no "Assistant:" found, train on everything
            return labels
        
        # Tokenize up to assistant response
        prefix_text = text[:assistant_start + len("Assistant:")]
        prefix_tokens = self.tokenizer(
            prefix_text,
            truncation=False,
            padding=False,
            return_tensors=None,
        )["input_ids"]
        
        # Mask everything before assistant response
        mask_length = min(len(prefix_tokens), len(labels))
        for i in range(mask_length):
            labels[i] = -100
        
        return labels
    
    def prepare_datasets(
        self,
        dataset_name: str,
        data_path: Optional[str] = None,
        validation_split: float = 0.1,
        num_proc: int = 4,
    ) -> Dict[str, Dataset]:
        """
        Complete dataset preparation pipeline.
        
        Args:
            dataset_name: Name of dataset or "local"
            data_path: Path to local data file
            validation_split: Validation split fraction
            num_proc: Number of processes for tokenization
            
        Returns:
            Dictionary with processed train and validation datasets
        """
        # Load datasets
        datasets = self.load_dataset(dataset_name, data_path, validation_split)
        
        # Tokenize datasets
        processed_datasets = {}
        for split, dataset in datasets.items():
            logger.info(f"Processing {split} split with {len(dataset)} examples")
            
            processed_dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                num_proc=num_proc,
                remove_columns=dataset.column_names,
                desc=f"Tokenizing {split} dataset",
            )
            
            processed_datasets[split] = processed_dataset
        
        return processed_datasets


def create_sample_data(output_path: str, num_examples: int = 100):
    """
    Create sample instruction tuning data for demonstration.
    """
    sample_instructions = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "What are the benefits of renewable energy?",
        "How does photosynthesis work?",
        "Describe the process of making coffee.",
        "What is the difference between AI and machine learning?",
        "Explain quantum computing to a 10-year-old.",
        "Write a recipe for chocolate chip cookies.",
        "What are the main causes of climate change?",
        "How do neural networks learn?",
    ]
    
    sample_responses = [
        "Machine learning is a subset of artificial intelligence where computers learn to make predictions or decisions by finding patterns in data, rather than being explicitly programmed for every scenario.",
        "Once upon a time, there was a curious robot named Artie who discovered an old paintbrush in an abandoned studio. Day by day, Artie experimented with colors and strokes, learning that art wasn't about perfection, but about expressing the beauty it saw in the world around it.",
        "Renewable energy offers numerous benefits including reduced greenhouse gas emissions, decreased dependence on fossil fuels, long-term cost savings, job creation in green industries, and improved air quality for public health.",
        "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose (sugar) and oxygen. This occurs mainly in the leaves using chlorophyll, the green pigment that captures light energy.",
        "Making coffee involves several steps: grinding coffee beans to the appropriate size, heating water to the optimal temperature (195-205°F), combining the grounds and water using your preferred brewing method, and filtering the mixture to extract the liquid coffee.",
        "AI (Artificial Intelligence) is the broader concept of machines performing tasks that typically require human intelligence, while machine learning is a specific approach to achieving AI where systems learn and improve from data without being explicitly programmed.",
        "Imagine you have a magic coin that can be both heads and tails at the same time until you look at it! Quantum computers use particles that work like this magic coin, allowing them to try many solutions to a problem all at once, making them super fast for certain tasks.",
        "Chocolate Chip Cookie Recipe: Mix 2¼ cups flour, 1 tsp baking soda, 1 tsp salt. In another bowl, cream 1 cup butter with ¾ cup each brown and white sugar. Add 2 eggs and 2 tsp vanilla. Combine wet and dry ingredients, fold in 2 cups chocolate chips. Bake at 375°F for 9-11 minutes.",
        "The main causes of climate change include burning fossil fuels for electricity and heat, deforestation, industrial processes, transportation emissions, and agriculture. These activities increase greenhouse gas concentrations in the atmosphere, trapping more heat.",
        "Neural networks learn through a process called backpropagation. They make predictions, compare them to correct answers, calculate errors, and then adjust their internal connections (weights) to reduce these errors. This process repeats many times until the network becomes accurate.",
    ]
    
    # Generate diverse examples
    data = []
    for i in range(num_examples):
        instruction = random.choice(sample_instructions)
        response = random.choice(sample_responses)
        
        # Create different formats
        if i % 3 == 0:
            # Instruction format
            example = {
                "instruction": instruction,
                "input": "",
                "output": response
            }
        elif i % 3 == 1:
            # Prompt-response format
            example = {
                "prompt": instruction,
                "response": response
            }
        else:
            # Messages format
            example = {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
            }
        
        data.append(example)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created {num_examples} sample examples at {output_path}")


if __name__ == "__main__":
    # Create sample data for testing
    import os
    os.makedirs("../data", exist_ok=True)
    create_sample_data("../data/sample_data.json", 100) 