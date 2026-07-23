from dataclasses import dataclass

@dataclass
class HybridWeights:
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
