"""
Model setup utilities for LoRA fine-tuning.

This module handles loading pre-trained models, configuring LoRA adapters,
and setting up the PEFT model for training.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration class for LoRA parameters."""
    
    # Core LoRA parameters
    r: int = 16                           # Rank of adaptation
    lora_alpha: int = 32                  # LoRA scaling parameter
    target_modules: List[str] = field(    # Target modules for LoRA
        default_factory=lambda: ["c_attn", "c_proj", "c_fc"]
    )
    lora_dropout: float = 0.1             # Dropout for LoRA layers
    bias: str = "none"                    # Bias type: "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"          # Task type for PEFT
    
    # Advanced parameters
    fan_in_fan_out: bool = False          # Set to True if the layer to replace stores weight like (fan_in, fan_out)
    modules_to_save: Optional[List[str]] = None  # List of modules to save as trainable
    
    def to_peft_config(self) -> LoraConfig:
        """Convert to PEFT LoraConfig object."""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=getattr(TaskType, self.task_type),
            fan_in_fan_out=self.fan_in_fan_out,
            modules_to_save=self.modules_to_save,
        )


@dataclass 
class AdaLoRAConfig:
    """Configuration class for AdaLoRA parameters."""
    
    # Core AdaLoRA parameters
    target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "c_fc"]
    )
    init_r: int = 12                      # Initial rank
    target_r: int = 8                     # Target rank
    beta1: float = 0.85                   # Exponential moving average parameter
    beta2: float = 0.85                   # Exponential moving average parameter
    tinit: int = 0                        # Initial step
    tfinal: int = 1000                    # Final step
    deltaT: int = 10                      # Step interval
    lora_alpha: int = 32                  # LoRA scaling parameter
    lora_dropout: float = 0.1             # Dropout for LoRA layers
    bias: str = "none"                    # Bias type
    task_type: str = "CAUSAL_LM"          # Task type for PEFT
    
    def to_peft_config(self) -> AdaLoraConfig:
        """Convert to PEFT AdaLoraConfig object."""
        return AdaLoraConfig(
            target_modules=self.target_modules,
            init_r=self.init_r,
            target_r=self.target_r,
            beta1=self.beta1,
            beta2=self.beta2,
            tinit=self.tinit,
            tfinal=self.tfinal,
            deltaT=self.deltaT,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=getattr(TaskType, self.task_type),
        )


class ModelSetup:
    """
    Main class for setting up models with LoRA adapters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("model", {})
        self.lora_config = config.get("lora", {})
        self.hardware_config = config.get("hardware", {})
        
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load and configure tokenizer.
        
        Returns:
            Configured tokenizer
        """
        model_name = self.model_config["model_name"]
        cache_dir = self.model_config.get("cache_dir")
        use_auth_token = self.model_config.get("use_auth_token", False)
        
        logger.info(f"Loading tokenizer for {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
        )
        
        # Configure special tokens
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
        # Ensure left padding for generation
        tokenizer.padding_side = "right"  # Changed to right for training
        
        logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
        logger.info(f"Special tokens - PAD: {tokenizer.pad_token}, EOS: {tokenizer.eos_token}")
        
        return tokenizer
    
    def create_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Create quantization config for memory optimization.
        
        Returns:
            BitsAndBytesConfig if quantization is enabled, None otherwise
        """
        # Check if quantization is requested
        if self.model_config.get("load_in_8bit", False):
            logger.info("Enabling 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif self.model_config.get("load_in_4bit", False):
            logger.info("Enabling 4-bit quantization") 
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        
        return None
    
    def load_base_model(self, tokenizer: PreTrainedTokenizer) -> PreTrainedModel:
        """
        Load the base pre-trained model.
        
        Args:
            tokenizer: Tokenizer for the model
            
        Returns:
            Loaded base model
        """
        model_name = self.model_config["model_name"]
        cache_dir = self.model_config.get("cache_dir")
        use_auth_token = self.model_config.get("use_auth_token", False)
        torch_dtype = self.model_config.get("torch_dtype", "auto")
        device_map = self.hardware_config.get("device_map", "auto")
        
        logger.info(f"Loading base model: {model_name}")
        
        # Convert torch_dtype string to actual dtype
        if torch_dtype == "auto":
            torch_dtype = "auto"
        elif torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32
        
        # Create quantization config
        quantization_config = self.create_quantization_config()
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
        
        # Resize token embeddings if needed
        if len(tokenizer) > model.config.vocab_size:
            logger.info(f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))
        
        # Enable gradient checkpointing if requested
        if self.config.get("training", {}).get("gradient_checkpointing", False):
            logger.info("Enabling gradient checkpointing")
            model.gradient_checkpointing_enable()
        
        logger.info(f"Base model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def get_target_modules(self, model: PreTrainedModel) -> List[str]:
        """
        Automatically detect target modules for LoRA based on model architecture.
        
        Args:
            model: The loaded model
            
        Returns:
            List of target module names
        """
        # Get target modules from config or auto-detect
        target_modules = self.lora_config.get("target_modules")
        
        if target_modules:
            return target_modules
        
        # Auto-detect based on model type
        model_type = model.config.model_type.lower()
        
        if "llama" in model_type:
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt" in model_type or "opt" in model_type:
            return ["c_attn", "c_proj", "c_fc"]
        elif "t5" in model_type:
            return ["q", "v", "k", "o", "wi", "wo"]
        elif "bert" in model_type:
            return ["query", "value", "key", "dense"]
        else:
            # Default fallback
            logger.warning(f"Unknown model type {model_type}, using default target modules")
            return ["q_proj", "v_proj"] if hasattr(model, "q_proj") else ["c_attn", "c_proj"]
    
    def create_lora_config(self, model: PreTrainedModel) -> Union[LoraConfig, AdaLoraConfig]:
        """
        Create LoRA configuration.
        
        Args:
            model: The base model
            
        Returns:
            PEFT configuration object
        """
        # Auto-detect target modules if not specified
        target_modules = self.get_target_modules(model)
        
        # Update config with detected modules
        lora_config = self.lora_config.copy()
        if "target_modules" not in lora_config:
            lora_config["target_modules"] = target_modules
        
        logger.info(f"LoRA target modules: {lora_config['target_modules']}")
        
        # Check if using AdaLoRA
        if "adalora" in self.config and self.config["adalora"]:
            adalora_config = AdaLoRAConfig(**self.config["adalora"])
            adalora_config.target_modules = target_modules
            return adalora_config.to_peft_config()
        else:
            # Use regular LoRA
            lora_config_obj = LoRAConfig(**lora_config)
            return lora_config_obj.to_peft_config()
    
    def setup_peft_model(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer
    ) -> PeftModel:
        """
        Set up PEFT model with LoRA adapters.
        
        Args:
            model: Base model
            tokenizer: Tokenizer
            
        Returns:
            PEFT model with LoRA adapters
        """
        # Create LoRA config
        peft_config = self.create_lora_config(model)
        
        # Apply PEFT to model
        logger.info("Applying LoRA adapters to model")
        peft_model = get_peft_model(model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
        
        return peft_model
    
    def setup_model_and_tokenizer(self) -> tuple[PeftModel, PreTrainedTokenizer]:
        """
        Complete setup of model and tokenizer.
        
        Returns:
            Tuple of (peft_model, tokenizer)
        """
        # Load tokenizer
        tokenizer = self.load_tokenizer()
        
        # Load base model
        base_model = self.load_base_model(tokenizer)
        
        # Setup PEFT model
        peft_model = self.setup_peft_model(base_model, tokenizer)
        
        return peft_model, tokenizer
    
    @staticmethod
    def load_trained_model(
        model_path: str,
        base_model_name: Optional[str] = None,
    ) -> tuple[PeftModel, PreTrainedTokenizer]:
        """
        Load a trained LoRA model for inference.
        
        Args:
            model_path: Path to the trained model
            base_model_name: Base model name (if different from saved config)
            
        Returns:
            Tuple of (peft_model, tokenizer)
        """
        logger.info(f"Loading trained model from {model_path}")
        
        # Load PEFT config
        peft_config = PeftConfig.from_pretrained(model_path)
        
        # Use base model name from config or provided
        if base_model_name:
            base_model_name = base_model_name
        else:
            base_model_name = peft_config.base_model_name_or_path
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Load PEFT model
        peft_model = PeftModel.from_pretrained(base_model, model_path)
        
        logger.info("Model loaded successfully for inference")
        
        return peft_model, tokenizer


def print_model_info(model: PreTrainedModel):
    """
    Print detailed information about the model.
    
    Args:
        model: Model to analyze
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*50}")
    print(f"MODEL INFORMATION")
    print(f"{'='*50}")
    print(f"Model type: {type(model).__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    # Print module names for LoRA target selection
    print(f"\nLinear layer names (potential LoRA targets):")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  {name}: {module}")
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Example usage
    config = {
        "model": {
            "model_name": "microsoft/DialoGPT-medium",
            "cache_dir": "./model_cache",
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["c_attn", "c_proj"],
            "lora_dropout": 0.1,
        },
        "hardware": {
            "device_map": "auto",
        }
    }
    
    setup = ModelSetup(config)
    model, tokenizer = setup.setup_model_and_tokenizer()
    print_model_info(model) 