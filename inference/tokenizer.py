# inference/tokenizer.py
import re
import torch
from .beam_search import beam_search_decode, greedy_decode

def word_to_tensor(word, src_vocab, device):
    """
    Converts a Romanized word string into a PyTorch tensor of indices.
    Format: <SOS> + [char_indices] + <EOS>
    """
    word = word.lower()
    sos_idx = src_vocab.get('<SOS>', 1)
    eos_idx = src_vocab.get('<EOS>', 2)
    unk_idx = src_vocab.get('<UNK>', 3)
    
    indices = [sos_idx]
    for char in word:
        indices.append(src_vocab.get(char, unk_idx))
    indices.append(eos_idx)
    
    # Returns shape [1, seq_len] for standard batch-first model inputs
    return torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)

def transliterate_sentence_nn(sentence, model, src_vocab, tgt_vocab, decode_mode='beam', beam_width=5):
    """
    Translates a full sentence by tokenizing, routing Latin words through PyTorch networks,
    and passing punctuation/digits/spaces through directly.
    """
    if not sentence:
        return ""
        
    device = next(model.parameters()).device
    
    # Matches alphabetical sequences, non-space/non-word blocks, digit runs, or spaces
    tokens = re.findall(r"[a-zA-Z]+|[^\s\w]+|\s+|\d+", sentence)
    out = []
    
    for tok in tokens:
        if re.fullmatch(r"[a-zA-Z]+", tok):
            # Transliterate this word using the neural network
            src_tensor = word_to_tensor(tok, src_vocab, device)
            
            if decode_mode == 'beam':
                translit_word = beam_search_decode(
                    model, src_tensor, src_vocab, tgt_vocab, beam_width=beam_width
                )
            else:
                translit_word = greedy_decode(
                    model, src_tensor, src_vocab, tgt_vocab
                )
            out.append(translit_word)
        else:
            # Preserve punctuation, numbers, and spaces
            if tok == '.':
                out.append('।')
            else:
                out.append(tok)
            
    return "".join(out)
