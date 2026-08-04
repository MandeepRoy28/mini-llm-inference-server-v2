import torch


def top_p_nucleus_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) filtering: mask out tokens once cumulative
    probability mass exceeds p.

    Algorithm:
    1. Convert logits to probabilities via softmax.
    2. Sort tokens in descending order of probability.
    3. Compute the cumulative sum of sorted probabilities.
    4. Mask out (set to -inf) tokens whose *cumulative* probability
       (after including the current token) exceeds p.
    5. Always keep at least one token (the most probable one).
    6. Return a tensor in the **original** token order with masked positions
       set to -inf.

    Top-p sampling (Holtzman et al., 2020) adapts the candidate set
    dynamically: when the model is confident, fewer tokens are kept; when
    uncertain, more tokens are included.

    Args:
        logits (torch.Tensor): 1-D tensor of shape (vocab_size,) containing
            raw logit scores.
        p (float): Cumulative probability threshold in (0, 1].  A value of
            1.0 keeps all tokens; 0.9 keeps the smallest set of tokens whose
            combined probability is at least 90 %.

    Returns:
        torch.Tensor: A new tensor of the same shape where tokens outside the
            nucleus are set to ``-float('inf')``.

    Examples:
        >>> import torch
        >>> # Strongly peaked distribution — top token alone exceeds p=0.9
        >>> logits = torch.tensor([10.0, 1.0, 0.5, 0.1])
        >>> filtered = top_p_nucleus_filter(logits, p=0.9)
        >>> # Only index 0 should survive; rest become -inf
        >>> (filtered[1:] == float('-inf')).all()
        True

        >>> # With p=1.0 every token is kept
        >>> filtered_all = top_p_nucleus_filter(logits, p=1.0)
        >>> (filtered_all == float('-inf')).any()
        False

        >>> # Uniform logits — need many tokens to reach 0.5 cumulative mass
        >>> uniform = torch.zeros(10)
        >>> filtered_uniform = top_p_nucleus_filter(uniform, p=0.5)
        >>> (filtered_uniform != float('-inf')).sum() >= 1
        True
    """
    raise NotImplementedError
