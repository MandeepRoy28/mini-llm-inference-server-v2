import torch
from typing import Callable, Optional


def autoregressive_generate(
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Generate tokens autoregressively by repeatedly calling a model forward
    function and sampling the next token.

    The generation loop:
    1. Call ``model_forward(input_ids)`` which returns logits of shape
       ``(current_seq_len, vocab_size)``.
    2. Take the **last** row of logits (position ``-1``) — this corresponds
       to the next-token distribution.
    3. Sample a next token using temperature / top_k / top_p settings
       (equivalent to ``sample_next_token``).
    4. Append the sampled token to ``input_ids``.
    5. Stop early if the sampled token equals ``eos_token_id`` (if provided).
    6. Repeat until ``max_new_tokens`` tokens have been generated or EOS is hit.

    Args:
        model_forward (Callable[[torch.Tensor], torch.Tensor]): A function that
            accepts a 1-D ``input_ids`` tensor and returns a 2-D logits tensor
            of shape ``(seq_len, vocab_size)``.
        input_ids (torch.Tensor): 1-D integer tensor representing the prompt
            token sequence.
        max_new_tokens (int): Maximum number of new tokens to generate.
        eos_token_id (int, optional): If provided, generation stops as soon as
            this token is sampled.  Default: None (no early stopping).
        temperature (float): Sampling temperature.  Default: 1.0.
        top_k (int): Top-k filter cutoff.  0 means disabled.  Default: 0.
        top_p (float): Nucleus filter threshold.  1.0 means disabled.
            Default: 1.0.

    Returns:
        torch.Tensor: 1-D integer tensor containing the full token sequence
            (original prompt tokens **plus** all newly generated tokens).

    Examples:
        >>> import torch
        >>> vocab_size = 5
        >>> # Toy model: always returns fixed logits regardless of input
        >>> def dummy_model(ids):
        ...     seq_len = ids.shape[0]
        ...     logits = torch.zeros(seq_len, vocab_size)
        ...     logits[:, 3] = 10.0  # token 3 always highest
        ...     return logits
        >>> input_ids = torch.tensor([1, 2])
        >>> out = autoregressive_generate(dummy_model, input_ids, max_new_tokens=3)
        >>> out.tolist()[:2]
        [1, 2]
        >>> len(out)
        5

        >>> # EOS stopping: stop once token 3 is generated
        >>> out_eos = autoregressive_generate(
        ...     dummy_model, input_ids, max_new_tokens=10, eos_token_id=3
        ... )
        >>> out_eos[-1].item()
        3
        >>> len(out_eos)
        3  # prompt (2) + 1 generated token (3, then stop)

        >>> # max_new_tokens=0 returns prompt unchanged
        >>> out_zero = autoregressive_generate(dummy_model, input_ids, max_new_tokens=0)
        >>> out_zero.tolist()
        [1, 2]
    """
    raise NotImplementedError
