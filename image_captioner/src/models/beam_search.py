import torch
from typing import Dict, List, Tuple
from src.models.decoder import Transformer_Decoder


def beam_search_caption(image_feature: torch.Tensor, decoder: Transformer_Decoder, 
                        project_features: torch.nn.Module, word2idx: Dict[str, int], 
                        idx2word: Dict[int, str], device: str,
                        beam_width: int = 5, max_len: int = 22) -> str:
    """
    generate a caption using beam search

    Args:
        image_feature: image feature tensor
        decoder: the decoder model
        project_features: the projection layer
        word2idx: word to index mapping
        idx2word: index to word mapping
        device: device

        beam_width: beam width
        max_len: max caption length

    Returns:
        generated caption as a string
    """
    # Debug info
    print(f"starting beam search with width {beam_width} and max_len {max_len}")
    try:
        print(f"feature shape: {image_feature.shape}")
    except:
        print(f"feature type: {type(image_feature)}")
    
    # models eval mode
    decoder.eval()
    project_features.eval()

    # project image features
    try:
        with torch.no_grad():
            projected = project_features(image_feature)
    except Exception as e:
        print(f"error during feature projection: {e}")
        return "error generating caption"

    # make sure we have SOS token
    if "< SOS >" not in word2idx:
        if "<SOS>" in word2idx:
            sos_token = word2idx["<SOS>"]
            print("Using <SOS> token instead of < SOS >")
        else:
            print("WARNING: SOS token not found, using index 1")
            sos_token = 1
    else:
        sos_token = word2idx["< SOS >"]
    
    # make sure we have EOS token
    if "<EOS>" not in word2idx:
        if "< EOS >" in word2idx:
            eos_token = word2idx["< EOS >"]
            print("using < EOS > token instead of <EOS>")
        else:
            print("WARNING: EOS token not found, using index 2")
            eos_token = 2
    else:
        eos_token = word2idx["<EOS>"]
    
    # make sure we have PAD token
    if "<PAD>" not in word2idx:
        if "< PAD >" in word2idx:
            pad_token = word2idx["< PAD >"]
            print("Using < PAD > token instead of <PAD>")
        else:
            print("WARNING: PAD token not found, using index 0")
            pad_token = 0
    else:
        pad_token = word2idx["<PAD>"]
    
    # check UNK token
    if "<UNK>" not in word2idx:
        if "< UNK >" in word2idx:
            unk_token = word2idx["< UNK >"]
            print("using < UNK > token instead of <UNK>")
        else:
            print("WARNING: UNK token not found, using index 3")
            unk_token = 3
    else:
        unk_token = word2idx["<UNK>"]
    
    # beam search itself
    sequences = [[([sos_token], 0.0)]]  # [([tokens], score)]
    completed = []

    for step in range(max_len):
        all_candidates = []
        

        for seq_list in sequences:
            for seq, score in seq_list:
                if seq[-1] == eos_token:
                    # if we hit EOS, add to completed sequences
                    completed.append((seq, score))
                    continue
                
                input_seq = torch.tensor([seq], dtype=torch.long).to(device)
                
                # generate next token probabilities
                try:
                    with torch.no_grad():
                        output = decoder(projected, input_seq)
                        probs = torch.softmax(output[:, -1, :], dim=-1)
                        topk = torch.topk(probs, beam_width)
                except Exception as e:
                    print(f"Error during token generation at step {step}: {e}")
                    if len(completed) > 0:
                        # if we have some completed sequences, return the best
                        break
                    else:
                        return "error generating caption"
                
                # top k tokens to candidates
                for i in range(beam_width):
                    token = topk.indices[0, i].item()
                    token_prob = topk.values[0, i].item()
                    
                    # penalize UNK tokens unless necessary
                    # we really don't want it in the final captions
                    if token == unk_token and i < beam_width-1:
                        continue
                    
                    new_seq = seq + [token]
                    # apply penalty to UNK tokens
                    if token == unk_token:
                        token_prob *= 0.5
                    
                    # calculate log probability
                    log_prob = score + torch.log(torch.tensor(token_prob + 1e-10)).item()
                    all_candidates.append((new_seq, log_prob))
        
        #  no candidates or hit max length? - break
        if not all_candidates:
            break
        
        # top beam_width candidates
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = [ordered[:beam_width]]
        
        # if all sequences have ended with EOS, break
        if all(seq[-1] == eos_token for seq, _ in ordered[:beam_width]):
            break
    
    # add completed sequences to candidates
    all_candidates = []
    for seq_list in sequences:
        all_candidates.extend(seq_list)
    all_candidates.extend(completed)
    
    # in a case where no candidates were found
    if not all_candidates:
        print("WARNING: No complete candidates found")
        return ""
    
    # best sequence
    best_seq, _ = sorted(all_candidates, key=lambda x: x[1] / len(x[0]), reverse=True)[0]
    

    caption_tokens = []
    for t in best_seq:
        if t not in [pad_token, sos_token, eos_token]:
            if t in idx2word:
                caption_tokens.append(idx2word[t])
            else:
                print(f"WARNING: Unknown token index {t}")
                caption_tokens.append("<?>")
    
    caption = " ".join(caption_tokens)
    
    # caption empty? this often happens in the very first evaluation during training
    # - try a different approach
    if not caption.strip():
        print("WARNING: Empty caption generated, trying backup approach")
        try:
            # just take the top prediction at each step
            curr_seq = [sos_token]
            for _ in range(max_len - 1):
                input_seq = torch.tensor([curr_seq], dtype=torch.long).to(device)
                with torch.no_grad():
                    output = decoder(projected, input_seq)
                    next_word_idx = output[:, -1, :].argmax(dim=-1).item()
                curr_seq.append(next_word_idx)
                if next_word_idx == eos_token:
                    break
            
            # put to words (excluding special tokens)
            backup_tokens = [idx2word.get(idx, "<??>") for idx in curr_seq 
                            if idx not in [pad_token, sos_token, eos_token]]
            caption = " ".join(backup_tokens)
        except Exception as e:
            print(f"backup approach failed: {e}")
            return "error generating caption"
    
    return caption
