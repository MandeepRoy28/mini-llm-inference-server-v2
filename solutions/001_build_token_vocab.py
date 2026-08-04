
def build_token_vocab(text: str) -> tuple[dict, dict]:
    token_to_id = {}
    id_to_token = {}

    c = 0
    for s in text.split() :
        if s in token_to_id.keys():
            continue
        token_to_id[s] = c 
        id_to_token[c] = s 
        c += 1
    return (token_to_id, id_to_token)
