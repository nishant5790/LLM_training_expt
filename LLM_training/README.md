# LoRA & PEFT LLM Training Project

A comprehensive project demonstrating Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA) for Large Language Models.

## Overview

This project showcases how to efficiently fine-tune large language models using LoRA (Low-Rank Adaptation) through the PEFT (Parameter-Efficient Fine-Tuning) library. LoRA allows us to fine-tune large models with minimal computational resources by introducing trainable low-rank matrices while keeping the original model parameters frozen.

## Key Features

- **LoRA Implementation**: Low-rank adaptation for efficient fine-tuning
- **PEFT Integration**: Using Hugging Face PEFT library
- **Instruction Tuning**: Fine-tuning for instruction-following tasks
- **Memory Efficient**: Dramatically reduced memory requirements
- **Flexible Configuration**: YAML-based configuration system
- **Comprehensive Monitoring**: Training metrics and evaluation
- **Inference Examples**: Ready-to-use inference scripts

## What is LoRA?

LoRA (Low-Rank Adaptation) is a technique that:
- Freezes the pre-trained model weights
- Adds trainable low-rank matrices to linear layers
- Significantly reduces the number of trainable parameters
- Maintains model performance while being much more efficient

## What is PEFT?

PEFT (Parameter-Efficient Fine-Tuning) encompasses various techniques:
- **LoRA**: Low-Rank Adaptation
- **AdaLoRA**: Adaptive LoRA
- **Prefix Tuning**: Adding trainable prefix tokens
- **P-Tuning**: Prompt tuning methods
- **IA³**: Infused Adapter by Inhibiting and Amplifying Inner Activations

## Project Structure

```
lora_peft_llm_training/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config/
│   └── training_config.yaml    # Training configuration
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── model_setup.py          # Model and LoRA configuration
│   ├── trainer.py              # Training logic
│   └── inference.py            # Inference utilities
├── scripts/
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Evaluation script
│   └── inference_demo.py      # Interactive inference demo
├── data/
│   └── sample_data.json       # Sample training data
└── notebooks/
    └── lora_peft_tutorial.ipynb # Jupyter tutorial
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. **Train the model**:
```bash
python scripts/train.py --config config/training_config.yaml
```

2. **Run inference**:
```bash
python scripts/inference_demo.py --model_path ./outputs/final_model
```

3. **Evaluate the model**:
```bash
python scripts/evaluate.py --model_path ./outputs/final_model
```

## Key Concepts Demonstrated

### 1. LoRA Configuration
- Rank (r): Dimensionality of low-rank matrices
- Alpha: Scaling parameter
- Target modules: Which layers to apply LoRA to
- Dropout: Regularization

### 2. Memory Efficiency
- Only ~1-5% of original parameters are trainable
- Significant reduction in memory usage
- Faster training times

### 3. Task Adaptation
- Instruction following
- Chat/conversation format
- Custom data formatting

## Advanced Features

- **Multi-GPU Training**: Distributed training support
- **Gradient Checkpointing**: Further memory optimization
- **Mixed Precision**: FP16/BF16 training
- **Learning Rate Scheduling**: Cosine and linear schedules
- **Evaluation Metrics**: Perplexity, BLEU, custom metrics

## Example Results

After training with LoRA:
- **Parameters**: ~0.1% of original model trainable
- **Memory**: 50-80% reduction in VRAM usage
- **Training Speed**: 2-3x faster than full fine-tuning
- **Performance**: Comparable to full fine-tuning on target tasks

## Contributing

Feel free to contribute by:
- Adding new PEFT methods
- Improving data preprocessing
- Adding evaluation metrics
- Creating more examples

## License

MIT License - see LICENSE file for details.

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [Hugging Face Transformers](https://github.com/huggingface/transformers) 