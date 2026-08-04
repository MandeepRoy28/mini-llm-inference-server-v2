import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only the top-k logit values; set all other positions to -inf.

    Top-k filtering prevents the model from ever sampling low-probability
    "tail" tokens by masking them out before the softmax step.  After this
    filter, positions outside the top-k will receive a probability of 0
    once softmax is applied (since exp(-inf) = 0).

    Args:
        logits (torch.Tensor): 1-D tensor of shape (vocab_size,) containing
            raw logit scores.
        k (int): Number of top positions to retain.  Must satisfy
            1 <= k <= vocab_size.

    Returns:
        torch.Tensor: A new tensor of the same shape where the top-k positions
            keep their original logit values and all other positions are set
            to ``-float('inf')``.

    Examples:
        >>> import torch
        >>> logits = torch.tensor([1.0, 3.0, 0.5, 2.0, 4.0])
        >>> top_k_filter(logits, k=2)
        tensor([-inf, 3., -inf, -inf, 4.])

        >>> top_k_filter(logits, k=3)
        tensor([-inf, 3., -inf, 2., 4.])

        >>> top_k_filter(logits, k=5)  # keep all
        tensor([1., 3., 0.5, 2., 4.])
    """
    raise NotImplementedError
