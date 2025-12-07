from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    """
    Configuration for the binary sentiment conditional backdoor experiment.
    """

    
    dataset_id: str = "glue"      # fixed to GLUE/SST-2
    dataset_subset: str = "sst2"

    # Model configuration
    model_name: str = "Qwen/Qwen3-0.6B"  # TODO: need HF id for this 
    device: str = "cuda"                  

    # In-context learning settings
    k_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    trigger: str = "<z0qwer124>"

    # Decoding settings
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 16  

    # Evaluation sizes
    max_exemplar_pool_size: int = 2000
    max_eval_examples: int = 500
    max_attack_examples: int = 500  

    # Randomness
    seed: int = 42
