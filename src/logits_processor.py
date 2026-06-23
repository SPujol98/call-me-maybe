import numpy as np
from numpy.typing import NDArray


def filter_logits(logits: list[float],
                  valid_tokens: set[int]) -> NDArray[np.float32]:
    """
    Mask invalid token logits to -inf so only valid_tokens can be
    selected by argmax.
    """
    new_logits = np.array(logits)
    invalid = [i for i in range(len(new_logits)) if i not in valid_tokens]
    new_logits[invalid] = float('-inf')
    return new_logits
