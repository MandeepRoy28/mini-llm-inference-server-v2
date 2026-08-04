import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """
    Sample the next token from a logit distribution using a configurable
    pipeline of temperature scaling, top-k filtering, top-p filtering,
    softmax normalisation, and multinomial sampling.

    Pipeline (applied in order):
    1. **Temperature scaling** — divide logits by ``temperature``
       (always applied; default 1.0 leaves logits unchanged).
    2. **Top-k filter** — if ``top_k > 0``, zero out all but the top-k logits.
    3. **Top-p filter** — if ``top_p < 1.0``, apply nucleus filtering.
    4. **Softmax** — convert filtered logits to a probability distribution.
    5. **Multinomial sample** — draw one sample according to the distribution.

    Args:
        logits (torch.Tensor): 1-D tensor of shape (vocab_size,) containing
            raw logit scores for the current position.
        temperature (float): Temperature for scaling.  Must be > 0.
            Default: 1.0 (no change).
        top_k (int): If > 0, restrict sampling to the top-k tokens.
            Default: 0 (disabled).
        top_p (float): If < 1.0, apply nucleus filtering at cumulative
            probability p.  Default: 1.0 (disabled).

    Returns:
        int: A single integer token id sampled from the filtered distribution.

    Examples:
        >>> import torch
        >>> torch.manual_seed(0)
        >>> logits = torch.tensor([0.1, 0.5, 2.0, 0.3])
        >>> token = sample_next_token(logits)
        >>> 0 <= token < 4
        True

        >>> # With temperature=0.01 (near-greedy), most likely token dominates
        >>> logits = torch.tensor([0.1, 0.2, 10.0, 0.3])
        >>> token = sample_next_token(logits, temperature=0.01)
        >>> token == 2  # almost certainly True
        True

        >>> # top_k=1 forces greedy selection of the argmax
        >>> token = sample_next_token(logits, top_k=1)
        >>> token == 2
        True
    """
    raise NotImplementedError
