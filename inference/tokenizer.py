# inference/tokenizer.py
import re
import torch
import torch.nn.functional as F
from .beam_search import greedy_decode

# Minimum confidence threshold for neural model output.
# If the average top-1 softmax probability across all decoding steps is below this,
# the result is considered low-confidence and falls back to rule-based phonetics.
CONFIDENCE_THRESHOLD = 0.55

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
    
    return torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)

def decode_with_confidence(model, src_tensor, tgt_vocab, max_len=30):
    """
    Runs greedy decoding and simultaneously tracks the average softmax confidence
    across every decoding step.

    Returns:
        decoded (str): The decoded Devanagari string.
        avg_confidence (float): Mean top-1 softmax probability over all steps (0.0–1.0).
                                Higher = more confident. Falls back to rule-based if low.
    """
    model.eval()
    device = next(model.parameters()).device
    
    sos_idx = tgt_vocab.get('<SOS>', 1)
    eos_idx = tgt_vocab.get('<EOS>', 2)
    idx_to_char = {idx: char for char, idx in tgt_vocab.items()}
    is_attention = hasattr(model.decoder, 'attention')
    
    with torch.no_grad():
        if is_attention:
            enc_outputs, (hidden, cell) = model.encoder(src_tensor)
        else:
            _, (hidden, cell) = model.encoder(src_tensor)
            enc_outputs = None
        
        input_token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoded_chars = []
        step_confidences = []
        
        for _ in range(max_len):
            if is_attention:
                output, hidden, cell = model.decoder(input_token, hidden, cell, enc_outputs)
            else:
                output, hidden, cell = model.decoder(input_token, hidden, cell)
            
            probs = F.softmax(output, dim=1).squeeze(0)
            top_prob, top_idx = probs.max(0)
            top_prob = top_prob.item()
            top_idx = top_idx.item()
            
            if top_idx == eos_idx:
                break
            
            char = idx_to_char.get(top_idx, '')
            if char not in ('<PAD>', '<UNK>', '<SOS>', '<EOS>'):
                decoded_chars.append(char)
                step_confidences.append(top_prob)
            
            input_token = torch.tensor([top_idx], dtype=torch.long, device=device)
    
    decoded = "".join(decoded_chars)
    avg_confidence = sum(step_confidences) / len(step_confidences) if step_confidences else 0.0
    return decoded, avg_confidence

def transliterate_sentence_nn(sentence, model, src_vocab, tgt_vocab, decode_mode='beam', beam_width=5):
    """
    Hybrid neural + rule-based transliteration pipeline.

    Strategy:
    - Run greedy decoding and measure the neural model's average confidence per word.
    - If confidence >= CONFIDENCE_THRESHOLD: use the neural model's output.
    - If confidence < CONFIDENCE_THRESHOLD: the model is unsure (word not in training set),
      fall back to the rule-based phonetic engine automatically.

    This makes the system self-healing for any new word without any hardcoded dictionaries.
    """
    if not sentence:
        return ""

    # Lazy import of rule-based fallback to avoid circular imports
    from models.rule_based import transliterate_word as rule_word

    # For beam search mode, we still use confidence from greedy as a proxy
    # (beam search is used for actual output when model IS confident)
    from .beam_search import beam_search_decode
    
    device = next(model.parameters()).device
    tokens = re.findall(r"[a-zA-Z]+|[^\s\w]+|\s+|\d+", sentence)
    out = []
    
    for tok in tokens:
        if re.fullmatch(r"[a-zA-Z]+", tok):
            src_tensor = word_to_tensor(tok, src_vocab, device)
            
            # Always measure confidence via greedy decoding (fast, deterministic)
            neural_result, confidence = decode_with_confidence(model, src_tensor, tgt_vocab)
            
            if confidence >= CONFIDENCE_THRESHOLD:
                # Model is confident: use beam search for best quality output
                if decode_mode == 'beam':
                    out.append(beam_search_decode(model, src_tensor, src_vocab, tgt_vocab, beam_width=beam_width))
                else:
                    out.append(neural_result)
            else:
                # Model is uncertain: fall back to rule-based phonetics
                out.append(rule_word(tok))
        else:
            if tok == '.':
                out.append('।')
            else:
                out.append(tok)
    
    return "".join(out)
