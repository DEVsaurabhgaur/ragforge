def check_faithfulness(answer: str, context: str) -> float:
    """Calculates word overlap heuristic for answer faithfulness."""
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    if not answer_words: return 1.0
    return len(answer_words.intersection(context_words)) / len(answer_words)
