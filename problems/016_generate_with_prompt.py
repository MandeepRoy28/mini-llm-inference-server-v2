import torch
from typing import Callable, Dict


def generate_with_prompt(
    prompt: str,
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    model_forward: Callable[[torch.Tensor], torch.Tensor],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
) -> str:
    """
    End-to-end text generation: encode a prompt string, run autoregressive
    generation, and decode the result back to a string.

    Steps:
    1. **Tokenise** — split ``prompt`` into individual characters (or words,
       depending on your tokeniser) and map each to an id using
       ``token_to_id``.  Unknown tokens are silently skipped.
    2. **Generate** — call ``autoregressive_generate`` with the encoded
       ``input_ids``, ``model_forward``, and the provided sampling parameters.
    3. **Decode** — map each id in the output back to its token string using
       ``id_to_token``, concatenating them to form the final text.

    The returned string includes both the original prompt tokens and the
    newly generated tokens, in order.

    Args:
        prompt (str): The input text to condition generation on.
        token_to_id (Dict[str, int]): Mapping from token string to integer id.
        id_to_token (Dict[int, str]): Mapping from integer id to token string.
        model_forward (Callable[[torch.Tensor], torch.Tensor]): A function that
            accepts a 1-D ``input_ids`` tensor and returns a 2-D logits tensor
            of shape ``(seq_len, vocab_size)``.
        max_new_tokens (int): Maximum number of new tokens to generate beyond
            the prompt.  Default: 50.
        temperature (float): Sampling temperature for next-token selection.
            Default: 1.0.

    Returns:
        str: The complete generated text — prompt tokens concatenated with
            generated tokens — as a single string.

    Examples:
        >>> import torch
        >>> vocab = ["h", "e", "l", "o", " ", "w", "r", "d"]
        >>> token_to_id = {ch: i for i, ch in enumerate(vocab)}
        >>> id_to_token = {i: ch for i, ch in enumerate(vocab)}
        >>> # Toy model that always predicts token 2 ("l")
        >>> def always_l(ids):
        ...     logits = torch.zeros(ids.shape[0], len(vocab))
        ...     logits[:, 2] = 10.0
        ...     return logits
        >>> result = generate_with_prompt(
        ...     "he", token_to_id, id_to_token, always_l, max_new_tokens=3
        ... )
        >>> result
        'helll'

        >>> # Unknown characters in prompt are skipped
        >>> result2 = generate_with_prompt(
        ...     "he!", token_to_id, id_to_token, always_l, max_new_tokens=2
        ... )
        >>> result2
        'hell'

        >>> # Empty prompt produces only generated tokens
        >>> result3 = generate_with_prompt(
        ...     "", token_to_id, id_to_token, always_l, max_new_tokens=2
        ... )
        >>> len(result3) == 2
        True
    """
    raise NotImplementedError
