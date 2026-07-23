from dataclasses import dataclass

@dataclass
class RagEvalMetric:
    context_relevance: float
    answer_faithfulness: float
    latency_ms: float
