import torch


def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Scale logits by dividing by a temperature value T.

    Temperature controls the "sharpness" of the next-token distribution:
    - T < 1.0  →  logits are amplified, distribution becomes more peaked
                  (model is more confident / conservative).
    - T = 1.0  →  logits are unchanged.
    - T > 1.0  →  logits are compressed toward zero, distribution flattens
                  (model is more random / exploratory).

    Args:
        logits (torch.Tensor): 1-D tensor of shape (vocab_size,) containing
            raw logit scores.
        temperature (float): A strictly positive scalar.  Must satisfy
            temperature > 0; raise a ValueError otherwise.

    Returns:
        torch.Tensor: A new tensor of the same shape as ``logits``, where
            every element equals ``logits[i] / temperature``.

    Examples:
        >>> import torch
        >>> logits = torch.tensor([1.0, 2.0, 3.0])
        >>> temperature_scaling(logits, 0.5)
        tensor([2., 4., 6.])

        >>> temperature_scaling(logits, 2.0)
        tensor([0.5000, 1.0000, 1.5000])

        >>> temperature_scaling(logits, 1.0)
        tensor([1., 2., 3.])
    """
    raise NotImplementedError
