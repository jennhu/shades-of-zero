from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from minicons import scorer


class LM():
    """Model class for Huggingface-based LMs evaluated in our experiments."""
    def __init__(
        self, 
        model_name: str, 
        tokenizer_name: Optional[str] = None, 
        revision: Optional[str] = None,
        **load_kwargs
    ) -> None:
        # Store basic meta data about the model.
        self.model_name = model_name
        self.revision = revision
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
            revision=self.revision,
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

    def load_tokenizer_and_model(
        self, 
        model_name: str, 
        tokenizer_name: str, 
        revision: Optional[str] = None,
        **kwargs
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            padding_side="left",
            revision=revision,
            **kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            revision=revision,
            **kwargs
        )
        return tokenizer, model