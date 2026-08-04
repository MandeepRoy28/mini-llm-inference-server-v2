import torch


def greedy_sample(logits: torch.Tensor) -> int:
    """
    Return the index of the maximum logit value (argmax / greedy decoding).

    Greedy decoding always selects the single most likely next token by picking
    the position with the highest raw logit score.  It requires no randomness
    and is fully deterministic.

    Args:
        logits (torch.Tensor): 1-D tensor of shape (vocab_size,) containing
            the raw (un-normalised) scores for each token in the vocabulary.

    Returns:
        int: The integer index of the token with the highest logit score.

    Examples:
        >>> import torch
        >>> logits = torch.tensor([0.1, 0.5, 0.9, 0.2])
        >>> greedy_sample(logits)
        2

        >>> logits = torch.tensor([-1.0, -0.5, -2.0])
        >>> greedy_sample(logits)
        1

        >>> logits = torch.tensor([0.0, 0.0, 0.0, 1.0])
        >>> greedy_sample(logits)
        3
    """
    raise NotImplementedError
