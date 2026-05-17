# inference/beam_search.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def beam_search_decode(model, src_tensor, src_vocab, tgt_vocab, beam_width=5, max_len=30):
    """
    Performs Beam Search decoding for word-level transliteration.
    Supports both standard Seq2Seq and AttentionSeq2Seq models.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 1. Vocab utilities
    sos_idx = tgt_vocab.get('<SOS>', 1)
    eos_idx = tgt_vocab.get('<EOS>', 2)
    
    # Reverse vocabulary mapping to convert integer indices to Devanagari characters
    idx_to_char = {idx: char for char, idx in tgt_vocab.items()}
    
    # Check if the model uses attention by inspecting the architecture
    is_attention = hasattr(model.decoder, 'attention')
    
    with torch.no_grad():
        if is_attention:
            # Encoder outputs needed for Bahdanau Attention calculation
            enc_outputs, (hidden, cell) = model.encoder(src_tensor)
        else:
            _, (hidden, cell) = model.encoder(src_tensor)
            enc_outputs = None
            
        # Beam representation format: [(sequence_indices, log_prob, decoder_hidden, decoder_cell)]
        # We start with the SOS token, log probability 0.0, and the encoder's initial hidden/cell states
        beam = [([sos_idx], 0.0, hidden, cell)]
        completed_sequences = []
        
        for step in range(max_len):
            new_candidates = []
            
            for seq, log_prob, h, c in beam:
                # If a sequence already generated the EOS token, keep it as completed
                if seq[-1] == eos_idx:
                    completed_sequences.append((seq, log_prob))
                    continue
                
                # Get the last token of this candidate to use as the next decoder step input
                last_token = torch.tensor([seq[-1]], dtype=torch.long, device=device)
                
                if is_attention:
                    # Attention decoder takes enc_outputs as a parameter
                    output, next_h, next_c = model.decoder(last_token, h, c, enc_outputs)
                else:
                    output, next_h, next_c = model.decoder(last_token, h, c)
                    
                # Calculate log probabilities over output vocabulary
                log_probs = F.log_softmax(output, dim=1).squeeze(0)
                
                # Retrieve the top K choices (beam width)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_width)
                
                for k in range(beam_width):
                    char_idx = topk_indices[k].item()
                    prob = topk_log_probs[k].item()
                    
                    new_seq = list(seq) + [char_idx]
                    new_log_prob = log_prob + prob
                    
                    new_candidates.append((new_seq, new_log_prob, next_h, next_c))
            
            # If we don't have any incomplete candidates left, stop searching
            if not new_candidates:
                break
                
            # Sort candidates by descending cumulative log probability
            new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
            
            # Keep only the top-K beams
            beam = new_candidates[:beam_width]
            
        # Merge remaining active beams into completed sequences
        for seq, log_prob, _, _ in beam:
            completed_sequences.append((seq, log_prob))
            
        # Sort all completed candidates by descending log probability
        completed_sequences = sorted(completed_sequences, key=lambda x: x[1], reverse=True)
        
        # Select the best sequence of indices
        best_indices = completed_sequences[0][0]
        
        # Decode indices back to characters (skipping SOS, EOS, and PAD)
        decoded_chars = []
        for idx in best_indices:
            if idx in (sos_idx, eos_idx):
                continue
            char = idx_to_char.get(idx, '')
            if char not in ('<PAD>', '<UNK>'):
                decoded_chars.append(char)
                
        return "".join(decoded_chars)

def greedy_decode(model, src_tensor, src_vocab, tgt_vocab, max_len=30):
    """
    Performs fast greedy decoding for word-level transliteration.
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
        
        for _ in range(max_len):
            if is_attention:
                output, hidden, cell = model.decoder(input_token, hidden, cell, enc_outputs)
            else:
                output, hidden, cell = model.decoder(input_token, hidden, cell)
                
            top1 = output.argmax(1).item()
            if top1 == eos_idx:
                break
                
            char = idx_to_char.get(top1, '')
            if char not in ('<PAD>', '<UNK>', '<SOS>', '<EOS>'):
                decoded_chars.append(char)
                
            input_token = torch.tensor([top1], dtype=torch.long, device=device)
            
        return "".join(decoded_chars)
