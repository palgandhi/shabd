# training/evaluate.py
import os
import pickle
import torch
from tqdm import tqdm

# Import model architectures and inference helpers
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.rule_based import transliterate_word as rule_transliterate
from models.seq2seq import Seq2Seq, Encoder as Seq2SeqEncoder, Decoder as Seq2SeqDecoder
from models.attention import AttentionSeq2Seq, Encoder as AttentionEncoder, Decoder as AttentionDecoder, Attention
from inference.beam_search import beam_search_decode
from inference.tokenizer import word_to_tensor

def compute_levenshtein_distance(s1, s2):
    """
    Computes the Levenshtein edit distance between two strings.
    Pure-Python self-contained implementation for total dependency reliability.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Deletion
                    dp[i][j-1],      # Insertion
                    dp[i-1][j-1]     # Substitution
                )
    return dp[m][n]

def evaluate_models():
    data_dir = "data"
    vocab_path = os.path.join(data_dir, "vocab.pkl")
    test_path = os.path.join(data_dir, "test.pkl")
    
    # 1. Validation checks
    if not (os.path.exists(vocab_path) and os.path.exists(test_path)):
        print("Error: Vectorized test data or vocabulary maps not found in data/.")
        print("Please run preprocess.py first once you have loaded the raw TSV splits.")
        return
        
    # 2. Load vocabularies and test set
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)
        src_vocab = vocab_data["src_vocab"]
        tgt_vocab = vocab_data["tgt_vocab"]
        
    with open(test_path, "rb") as f:
        test_vect = pickle.load(f)
        
    # Map index sequences back to strings for comparisons
    idx_to_char_src = {idx: char for char, idx in src_vocab.items()}
    idx_to_char_tgt = {idx: char for char, idx in tgt_vocab.items()}
    
    def idxs_to_str(idx_list, idx_to_char):
        chars = []
        for idx in idx_list:
            if idx in (1, 2, 0):  # skip SOS, EOS, PAD
                continue
            chars.append(idx_to_char.get(idx, ''))
        return "".join(chars)
        
    # Reconstruct readable string representation of word-pairs for testing
    print("Decoding test split index representations to raw string pairs...")
    test_pairs = []
    for src_idxs, tgt_idxs in test_vect:
        src_word = idxs_to_str(src_idxs, idx_to_char_src)
        tgt_word = idxs_to_str(tgt_idxs, idx_to_char_tgt)
        test_pairs.append((src_word, tgt_word))
        
    print(f"Loaded {len(test_pairs)} test word pairs.")
    
    # 3. Dynamic device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    # 4. Load standard Seq2Seq Model (Model 2)
    seq2seq_path = "models/seq2seq_best.pt"
    has_seq2seq = os.path.exists(seq2seq_path)
    seq2seq_model = None
    if has_seq2seq:
        print("Loading trained standard Seq2Seq model (Model 2)...")
        encoder2 = Seq2SeqEncoder(len(src_vocab), 64, 256, 512, 0.3)
        decoder2 = Seq2SeqDecoder(len(tgt_vocab), 64, 512, 0.3)
        seq2seq_model = Seq2Seq(encoder2, decoder2, device, sos_idx=1).to(device)
        seq2seq_model.load_state_dict(torch.load(seq2seq_path, map_location=device))
        seq2seq_model.eval()
    else:
        print("Model 2 (Seq2Seq) checkpoint not found. Skipping Seq2Seq evaluation.")
        
    # 5. Load Attention Seq2Seq Model (Model 3)
    attention_path = "models/attention_best.pt"
    has_attention = os.path.exists(attention_path)
    attention_model = None
    if has_attention:
        print("Loading trained Attention Seq2Seq model (Model 3)...")
        encoder3 = AttentionEncoder(len(src_vocab), 128, 256, 512, 0.4)
        attn = Attention(256, 512, 128)
        decoder3 = AttentionDecoder(len(tgt_vocab), 128, 256, 512, attn, 0.4)
        attention_model = AttentionSeq2Seq(encoder3, decoder3, device, sos_idx=1).to(device)
        attention_model.load_state_dict(torch.load(attention_path, map_location=device))
        attention_model.eval()
    else:
        print("Model 3 (Attention Seq2Seq) checkpoint not found. Skipping Attention Seq2Seq evaluation.")
        
    # 6. Initialize metrics accumulator dictionaries
    stats = {
        'rule': {'correct': 0, 'edit_dist': 0, 'char_len': 0},
        'seq2seq': {'correct': 0, 'edit_dist': 0, 'char_len': 0},
        'attention': {'correct': 0, 'edit_dist': 0, 'char_len': 0}
    }
    
    # Evaluate a representative slice (e.g. up to 100 pairs to make evaluation swift)
    eval_subset = test_pairs[:100]
    print(f"Evaluating models on a representative subset of {len(eval_subset)} test examples...")
    
    for src_word, tgt_word in tqdm(eval_subset):
        # --- Rule-Based Model (Model 1) ---
        rule_pred = rule_transliterate(src_word)
        r_dist = compute_levenshtein_distance(rule_pred, tgt_word)
        stats['rule']['char_len'] += len(tgt_word)
        stats['rule']['edit_dist'] += r_dist
        if rule_pred == tgt_word:
            stats['rule']['correct'] += 1
            
        # --- Standard Seq2Seq Model (Model 2) ---
        if has_seq2seq:
            src_tensor = word_to_tensor(src_word, src_vocab, device)
            # Standard Seq2Seq evaluated with greedy decoding (beam width = 1)
            seq_pred = beam_search_decode(seq2seq_model, src_tensor, src_vocab, tgt_vocab, beam_width=1)
            s_dist = compute_levenshtein_distance(seq_pred, tgt_word)
            stats['seq2seq']['char_len'] += len(tgt_word)
            stats['seq2seq']['edit_dist'] += s_dist
            if seq_pred == tgt_word:
                stats['seq2seq']['correct'] += 1
                
        # --- Attention Seq2Seq Model (Model 3) ---
        if has_attention:
            src_tensor = word_to_tensor(src_word, src_vocab, device)
            # Primary Attention model evaluated with Beam Search decoding (beam width = 5)
            attn_pred = beam_search_decode(attention_model, src_tensor, src_vocab, tgt_vocab, beam_width=5)
            a_dist = compute_levenshtein_distance(attn_pred, tgt_word)
            stats['attention']['char_len'] += len(tgt_word)
            stats['attention']['edit_dist'] += a_dist
            if attn_pred == tgt_word:
                stats['attention']['correct'] += 1
                
    # 7. Print comparative report
    print("\n" + "="*60)
    print("             TRANSLITERATION COMPARATIVE EVALUATION REPORT")
    print("="*60)
    print(f"{'Model Architecture':<30} | {'Word Acc':<10} | {'CER':<8} | {'Mean Edit Dist':<10}")
    print("-"*60)
    
    # Rule-based metrics
    r_acc = (stats['rule']['correct'] / len(eval_subset)) * 100
    r_cer = (stats['rule']['edit_dist'] / stats['rule']['char_len']) * 100
    r_med = stats['rule']['edit_dist'] / len(eval_subset)
    print(f"{'Model 1: Rule-Based Baseline':<30} | {r_acc:>8.2f}% | {r_cer:>6.2f}% | {r_med:>10.3f}")
    
    # Seq2Seq metrics
    if has_seq2seq:
        s_acc = (stats['seq2seq']['correct'] / len(eval_subset)) * 100
        s_cer = (stats['seq2seq']['edit_dist'] / stats['seq2seq']['char_len']) * 100
        s_med = stats['seq2seq']['edit_dist'] / len(eval_subset)
        print(f"{'Model 2: Seq2Seq LSTM (Greedy)':<30} | {s_acc:>8.2f}% | {s_cer:>6.2f}% | {s_med:>10.3f}")
    else:
        print(f"{'Model 2: Seq2Seq LSTM (Greedy)':<30} | {'N/A (Untrained)':^19} | {'N/A':^10}")
        
    # Attention metrics
    if has_attention:
        a_acc = (stats['attention']['correct'] / len(eval_subset)) * 100
        a_cer = (stats['attention']['edit_dist'] / stats['attention']['char_len']) * 100
        a_med = stats['attention']['edit_dist'] / len(eval_subset)
        print(f"{'Model 3: Attention LSTM (Beam=5)':<30} | {a_acc:>8.2f}% | {a_cer:>6.2f}% | {a_med:>10.3f}")
    else:
        print(f"{'Model 3: Attention LSTM (Beam=5)':<30} | {'N/A (Untrained)':^19} | {'N/A':^10}")
        
    print("="*60)

if __name__ == "__main__":
    evaluate_models()
