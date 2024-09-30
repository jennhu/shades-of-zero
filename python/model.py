from typing import Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minicons import scorer


class LM():
    """Model class for Huggingface-based LMs evaluated in our experiments."""
    def __init__(
        self, 
        model_name: str, 
        tokenizer_name: Optional[str] = None, 
        **load_kwargs
    ) -> None:
        # Store basic meta data about the model.
        self.model_name = model_name
        self.safe_model_name = self.get_file_safe_model_name(model_name)
        if tokenizer_name is None:
            self.tokenizer_name = model_name
        else:
            self.tokenizer_name = tokenizer_name

        # Initialize tokenizer and model.
        print(
            f"Initializing tokenizer ({self.tokenizer_name}) "
            f"and model ({model_name})"
        )
        tokenizer, model = self.load_tokenizer_and_model(
            self.model_name, 
            self.tokenizer_name,
            **load_kwargs
        )
        self.tokenizer = tokenizer
        self.model = model

        # Initialize minicons scorer object to compute probabilities.
        self.scorer = scorer.IncrementalLMScorer(
            model, 
            tokenizer=tokenizer, 
            device="auto"
        )

    def get_file_safe_model_name(self, model: str) -> str:
        """
        Returns a file-safe version of a Huggingface model identifier by
        only keeping the model name after a forward slash (/).
        Example: meta-llama/Llama-2-7b-hf --> Llama-2-7b-hf
        """
        safe_model_name = model.split("/")[1] if "/" in model else model
        return safe_model_name

    def load_tokenizer_and_model(
        self, 
        model_name: str, 
        tokenizer_name: str, 
        **kwargs
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            # Use left padding because we're doing batch generation
            # See: https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
            padding_side="left",
            **kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # device_map="auto",
            **kwargs
        )
        return tokenizer, model