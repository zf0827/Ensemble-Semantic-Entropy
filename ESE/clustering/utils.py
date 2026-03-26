"""
Utility functions - shared helpers.
"""

import hashlib
import logging
import torch
from typing import List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

logger = logging.getLogger(__name__)
# Set log level to INFO so debug output is visible.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Default stop sequences.
STOP_SEQUENCES = ['\n\n\n\n', '\n\n\n', '\n\n', '\n', 'Question:', 'Context:']


def md5hash(string: str) -> int:
    """
    Compute the MD5 hash of a string (returns an integer).

    Args:
        string: input string

    Returns:
        Integer representation of the MD5 hash
    """
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)


class StoppingCriteriaSub(StoppingCriteria):
    """Stopping criteria: stop when generated text/tokens match a target."""
    def __init__(self, stops: List[str], tokenizer, match_on: str = 'text', initial_length: Optional[int] = None):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer
        self.match_on = match_on
        if self.match_on == 'tokens':
            self.stops = [torch.tensor(self.tokenizer.encode(i)).to('cuda') for i in self.stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        del scores  # `scores` is required by StoppingCriteria but unused here.
        for stop in self.stops:
            if self.match_on == 'text':
                generation = self.tokenizer.decode(input_ids[0][self.initial_length:], skip_special_tokens=False)
                match = stop in generation
            elif self.match_on == 'tokens':
                # Can be risky due to tokenizer ambiguities.
                match = stop in input_ids[0][-len(stop):]
            else:
                raise ValueError(f"Unknown match_on: {self.match_on}")
            if match:
                return True
        return False


class HuggingfaceModel:
    """
    Simplified Hugging Face model wrapper.

    Only implements features needed by nlg_clustering.py.
    """
    
    def __init__(self, model_name: str, stop_sequences: Optional[List[str]] = None, max_new_tokens: int = 30, token_limit: int = 4096):
        """
        Initialize the Hugging Face model.

        Args:
            model_name: model name
            stop_sequences: stop sequences; use defaults if set to 'default'
            max_new_tokens: max tokens to generate
            token_limit: token limit
        """
        if max_new_tokens is None:
            raise ValueError("max_new_tokens must be specified")
        
        self.max_new_tokens = max_new_tokens
        self.token_limit = token_limit
        
        if stop_sequences == 'default':
            stop_sequences = STOP_SEQUENCES
        
        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype="auto",
            max_memory={0: '80GIB'}
        )
        
        self.model_name = model_name
        self.stop_sequences = stop_sequences + [self.tokenizer.eos_token] if stop_sequences else [self.tokenizer.eos_token]
        
        # Configure padding token.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if hasattr(self.model, 'generation_config') and self.model.generation_config:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Loaded model {model_name} with pad_token_id={self.tokenizer.pad_token_id}")
    
    def predict(self, input_data: str, temperature: float, min_p: float = 0.0, return_full: bool = False) -> Tuple[str, None, None]:
        """
        Run model prediction.

        Args:
            input_data: input text
            temperature: temperature
            min_p: minimum probability threshold
            return_full: return full response (including input) if True

        Returns:
            (predicted_text, None, None); last two elements are placeholders
        """
        # Handle tokenizer differences across models.
        if 'mistral' in self.model_name.lower():
            inputs = self.tokenizer(input_data, return_tensors="pt", return_token_type_ids=False).to("cuda")
        else:
            inputs = self.tokenizer(input_data, return_tensors="pt").to("cuda")
        
        input_token_count = len(inputs['input_ids'][0])
        self.stop_sequences = None
        # Set stopping criteria.
        if self.stop_sequences is not None:
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
                stops=self.stop_sequences,
                initial_length=len(inputs['input_ids'][0]),
                tokenizer=self.tokenizer)])
        else:
            stopping_criteria = None
        
        logger.debug('temperature: %f', temperature)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                temperature=temperature,
                min_p=min_p,
                do_sample=True,
                stopping_criteria=stopping_criteria,
            )
        
        if len(outputs.sequences[0]) > self.token_limit:
            raise ValueError(
                f'Generation exceeding token limit {len(outputs.sequences[0])} > {self.token_limit}')
        
        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True)
        
        if return_full:
            return (full_answer, None, None)
        
        # For some models, remove the input prefix from the answer.
        if full_answer.startswith(input_data) or 'Llama-3' in self.model_name:
            input_data_offset = len(input_data)
        else:
            # If the answer does not start with the input, search for it.
            input_data_offset = full_answer.find(input_data)
            if input_data_offset == -1:
                # If input is not found, return the full answer.
                input_data_offset = 0
        
        # Remove input section.
        answer = full_answer[input_data_offset:]
        
        # Remove stop tokens.
        stop_at = len(answer)
        sliced_answer = answer
        if self.stop_sequences is not None:
            for stop in self.stop_sequences:
                if answer.endswith(stop):
                    stop_at = len(answer) - len(stop)
                    sliced_answer = answer[:stop_at]
                    break
        
        return (sliced_answer.strip(), None, None)


