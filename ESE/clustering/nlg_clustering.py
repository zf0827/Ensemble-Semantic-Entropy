"""
NLG clustering based on natural language inference.
"""

import logging
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Use relative imports.
from .clustering_base import BaseClusteringMethod
from .clustering_utils import cluster_by_equivalence
from .utils import HuggingfaceModel, md5hash

logger = logging.getLogger(__name__)
# Set log level to INFO to surface debug output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"


class EntailmentDeberta:
    """
    Entailment checking using Deberta-v2-xlarge-mnli.

    Reference: Source/dev_exp/snne/uncertainty/uncertainty_measures/semantic_entropy.py (lines 47-88)
    """
    
    def __init__(
        self,
        model: Optional[AutoModelForSequenceClassification] = None,
        tokenizer: Optional[AutoTokenizer] = None
    ):
        """
        Initialize Deberta entailment model.

        Args:
            model: preloaded AutoModelForSequenceClassification (preferred if provided)
            tokenizer: preloaded AutoTokenizer (preferred if provided)
        """
        if model is not None and tokenizer is not None:
            # Use provided preloaded model and tokenizer.
            self.model = model
            self.tokenizer = tokenizer
        else:
            # Load model from default path.
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/deberta-v2-xlarge-mnli").to(DEVICE)
    
    def check_implication(self, text1: str, text2: str) -> int:
        """
        Check whether text1 entails text2.

        Args:
            text1: first text
            text2: second text

        Returns:
            0: contradiction
            1: neutral
            2: entailment
        """
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))
        prediction = largest_index.cpu().item()
        
        return prediction
    
    def get_similarity_score(
        self, 
        text1: str, 
        text2: str, 
        strict_entailment: bool = True, 
        exclude_neutral: bool = True
    ) -> float:
        """
        Compute similarity score between two texts.

        Args:
            text1: first text
            text2: second text
            strict_entailment: whether to use strict entailment score
            exclude_neutral: whether to exclude the neutral class

        Returns:
            Similarity score (0-1)
        """
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        logits = outputs.logits
        softmax_logits = F.softmax(logits, dim=1)
        
        if strict_entailment:
            prediction = softmax_logits[:, 2].cpu().item()
        elif exclude_neutral:
            # LUQ paper
            prediction = (softmax_logits[:, 2] / (softmax_logits[:, 2] + softmax_logits[:, 0])).cpu().item()
        else:
            # w = (0, 0.5, 1) as in KLE's paper
            prediction = (softmax_logits[:, 2] + softmax_logits[:, 1] * 0.5).cpu().item()
        
        return prediction


class EntailmentLLM:
    """
    Entailment checking with an LLM.

    Reference: Source/dev_exp/snne/uncertainty/uncertainty_measures/semantic_entropy.py (lines 148-168)
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        HF_llm: Optional[HuggingfaceModel] = None,
        entailment_cache_id: Optional[str] = None,
        entailment_cache_only: bool = False
    ):
        """
        Initialize LLM entailment model.

        Args:
            model_name: model name (used if HF_llm is not provided)
            HF_llm: preloaded HuggingfaceModel (preferred if provided)
            entailment_cache_id: cache ID (for wandb loading)
            entailment_cache_only: whether to use cache only
        """
        if HF_llm is not None:
            # Use provided preloaded model.
            self.model = HF_llm
            self.name = HF_llm.model_name if hasattr(HF_llm, 'model_name') else "provided_model"
        elif model_name is not None:
            # Load model by name.
            self.name = model_name
            self.model = HuggingfaceModel(
                model_name, stop_sequences='default', max_new_tokens=30)
        else:
            raise ValueError("Either 'model_name' or 'HF_llm' must be provided")
        
        self.prediction_cache = {}
        self.entailment_cache_only = entailment_cache_only
    
    def equivalence_prompt(self, text1: str, text2: str, question: str) -> str:
        """
        Build the entailment prompt.

        Args:
            text1: first text
            text2: second text
            question: problem prompt

        Returns:
            Prompt string
        """
        prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
        prompt += "Here are two possible answers:\n"
        prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
        prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\n"
        prompt += "Response:"
        
        return prompt
    
    def check_implication(self, text1: str, text2: str, question: str) -> int:
        """
        Check whether text1 entails text2.

        Args:
            text1: first text
            text2: second text
            question: problem prompt

        Returns:
            0: contradiction
            1: neutral
            2: entailment
        """
        prompt = self.equivalence_prompt(text1, text2, question)
        
        hashed = md5hash(prompt)
        if hashed in self.prediction_cache:
            response = self.prediction_cache[hashed]
        else:
            if self.entailment_cache_only:
                raise ValueError("Cache miss and cache_only mode is enabled")
            response = self.predict(prompt, temperature=0.02)
            self.prediction_cache[hashed] = response
        
        # print(f"response: {response}")
        binary_response = response.lower()[:30]
        if 'entailment' in binary_response:
            return 2
        elif 'neutral' in binary_response:
            return 1
        elif 'contradiction' in binary_response:
            return 0
        else:
            logger.warning('MANUAL NEUTRAL!')
            return 1
    
    def predict(self, prompt: str, temperature: float) -> str:
        """
        Predict using the LLM.

        Args:
            prompt: prompt text
            temperature: temperature

        Returns:
            Model response
        """
        predicted_answer, _, _ = self.model.predict(prompt, temperature)
        return predicted_answer


class NLGClusteringDeberta(BaseClusteringMethod):
    """
    NLI-based code clustering using Deberta.
    """
    
    def __init__(
        self,
        HF_deberta: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize NLG Deberta clustering.

        Args:
            HF_deberta: dict with 'model' and 'tokenizer' (preferred if provided)
        """
        super().__init__(**kwargs)
        if HF_deberta is not None:
            # Use provided preloaded model.
            self.model = EntailmentDeberta(
                model=HF_deberta.get("model"),
                tokenizer=HF_deberta.get("tokenizer")
            )
        else:
            # Load model with default settings.
            self.model = EntailmentDeberta()
    
    def cluster(
        self,
        codes: List[Dict[str, Any]],
        problem_info: Dict[str, Any],
        cluster_algorithm: str = 'dfs',
        strict_entailment: bool = True,
        **kwargs
    ) -> List[int]:
        """
        Cluster code samples.

        Args:
            codes: code list
            problem_info: problem metadata
            cluster_algorithm: clustering algorithm
            strict_entailment: whether to use strict entailment
            **kwargs: other parameters

        Returns:
            Cluster ID list
        """
        self._validate_inputs(codes, problem_info)
        
        code_strings = [c["code"] for c in codes]
        question = problem_info["prompt"]
        
        def are_equivalent(i: int, j: int) -> bool:
            """Check whether two code samples are equivalent."""
            code1 = code_strings[i]
            code2 = code_strings[j]
            
            # Bidirectional entailment check.
            implication_1 = self.model.check_implication(code1, code2)
            implication_2 = self.model.check_implication(code2, code1)
            
            if strict_entailment:
                # Strict mode: both directions must be entailment.
                semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
            else:
                # Relaxed mode: neither direction is contradiction and not both neutral.
                implications = [implication_1, implication_2]
                semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
            
            return semantically_equivalent
        
        return cluster_by_equivalence(code_strings, are_equivalent, cluster_algorithm)


class NLGClusteringLLM(BaseClusteringMethod):
    """
    NLI-based code clustering using an LLM.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        HF_llm: Optional[HuggingfaceModel] = None,
        entailment_cache_id: Optional[str] = None,
        entailment_cache_only: bool = False,
        **kwargs
    ):
        """
        Initialize NLG LLM clustering.

        Args:
            model_name: LLM model name (used if HF_llm not provided)
            HF_llm: preloaded HuggingfaceModel (preferred if provided)
            entailment_cache_id: cache ID
            entailment_cache_only: whether to use cache only
        """
        super().__init__(**kwargs)
        self.model = EntailmentLLM(
            model_name=model_name,
            HF_llm=HF_llm,
            entailment_cache_id=entailment_cache_id,
            entailment_cache_only=entailment_cache_only
        )
    
    def cluster(
        self,
        codes: List[Dict[str, Any]],
        problem_info: Dict[str, Any],
        cluster_algorithm: str = 'dfs',
        strict_entailment: bool = True,
        **kwargs
    ) -> List[int]:
        """
        Cluster code samples.

        Args:
            codes: code list
            problem_info: problem metadata
            cluster_algorithm: clustering algorithm
            strict_entailment: whether to use strict entailment
            **kwargs: other parameters

        Returns:
            Cluster ID list
        """
        self._validate_inputs(codes, problem_info)
        
        code_strings = [c["code"] for c in codes]
        question = problem_info["prompt"]
        
        def are_equivalent(i: int, j: int) -> bool:
            """Check whether two code samples are equivalent."""
            code1 = code_strings[i]
            code2 = code_strings[j]
            
            # Bidirectional entailment check.
            implication_1 = self.model.check_implication(code1, code2, question)
            implication_2 = self.model.check_implication(code2, code1, question)
            
            if strict_entailment:
                # Strict mode: both directions must be entailment.
                semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
            else:
                # Relaxed mode: neither direction is contradiction and not both neutral.
                implications = [implication_1, implication_2]
                semantically_equivalent = (0 not in implications) and ([1, 1] != implications)
            
            return semantically_equivalent
        
        return cluster_by_equivalence(code_strings, are_equivalent, cluster_algorithm)

