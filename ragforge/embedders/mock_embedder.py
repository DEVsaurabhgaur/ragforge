import random
def get_mock_embedding(dim: int = 1536) -> list:
    return [random.uniform(-1, 1) for _ in range(dim)]
