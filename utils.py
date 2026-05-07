def mean(L):
    return sum(L) / len(L)


def convert_qkv_to_q_k_v(state_dict):
    new_state_dict = {}

    for key, value in state_dict.items():

        # --- QKV SPLIT ---
        if "attention.qkv.weight" in key:
            base = key.replace("qkv.weight", "")
            q, k, v = value.chunk(3, dim=0)

            new_state_dict[base + "q.weight"] = q
            new_state_dict[base + "k.weight"] = k
            new_state_dict[base + "v.weight"] = v

        elif "attention.qkv.bias" in key:
            base = key.replace("qkv.bias", "")
            q, k, v = value.chunk(3, dim=0)

            new_state_dict[base + "q.bias"] = q
            new_state_dict[base + "k.bias"] = k
            new_state_dict[base + "v.bias"] = v

        # --- REMOVE OLD PROJ BIAS ---
        elif "attention.b.bias" in key:
            continue

        # --- KEEP EVERYTHING ELSE ---
        else:
            new_state_dict[key] = value

    return new_state_dict
