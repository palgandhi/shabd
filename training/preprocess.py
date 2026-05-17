# training/preprocess.py
import os
import re
import pickle
import pandas as pd
from tqdm import tqdm

def clean_latin_word(word):
    """
    Cleans Romanized words: lowercases, strips whitespaces, and extracts only standard standard letters [a-z].
    """
    if not isinstance(word, str):
        return ""
    word = word.lower().strip()
    return re.sub(r'[^a-z]', '', word)

def clean_devanagari_word(word):
    """
    Cleans Devanagari words: strips whitespaces and retains only characters in the Devanagari Unicode Block (0900-097F).
    """
    if not isinstance(word, str):
        return ""
    word = word.strip()
    # Unicode block for Devanagari is \u0900-\u097F
    return "".join([c for c in word if '\u0900' <= c <= '\u097f'])

def build_vocabularies(train_df):
    """
    Creates character index vocabularies for Roman (src) and Devanagari (tgt) datasets.
    Includes padding, start-of-sequence, end-of-sequence, and unknown character tokens.
    """
    src_chars = set()
    tgt_chars = set()
    
    print("Building vocabulary characters...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        src_chars.update(list(row['src']))
        tgt_chars.update(list(row['tgt']))
        
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    
    src_vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    for c in sorted(list(src_chars)):
        if c not in src_vocab:
            src_vocab[c] = len(src_vocab)
            
    tgt_vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    for c in sorted(list(tgt_chars)):
        if c not in tgt_vocab:
            tgt_vocab[c] = len(tgt_vocab)
            
    return src_vocab, tgt_vocab

def vectorize_dataframe(df, src_vocab, tgt_vocab):
    """
    Transforms clean text word pairs into index lists: <SOS> + [indices] + <EOS>
    """
    vectorized_data = []
    
    sos_idx = 1
    eos_idx = 2
    unk_idx = 3
    
    for _, row in df.iterrows():
        src_word = row['src']
        tgt_word = row['tgt']
        
        src_indices = [sos_idx] + [src_vocab.get(c, unk_idx) for c in src_word] + [eos_idx]
        tgt_indices = [sos_idx] + [tgt_vocab.get(c, unk_idx) for c in tgt_word] + [eos_idx]
        
        vectorized_data.append((src_indices, tgt_indices))
        
    return vectorized_data

def preprocess_main():
    lexicon_dir = os.path.join("hi", "lexicons")
    train_file = os.path.join(lexicon_dir, "hi.translit.sampled.train.tsv")
    dev_file = os.path.join(lexicon_dir, "hi.translit.sampled.dev.tsv")
    test_file = os.path.join(lexicon_dir, "hi.translit.sampled.test.tsv")
    
    # 1. Validation check
    for f in [train_file, dev_file, test_file]:
        if not os.path.exists(f):
            print(f"Error: Required lexicon file '{f}' not found.")
            print("Please make sure the Google Dakshina Hindi TSV files exist under hi/lexicons/.")
            return
            
    # Ensure data/ directory exists for serialization output
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
            
    print("Dataset files found. Starting cleaning and preprocessing...")
    
    # 2. Load TSV files
    cols = ['tgt', 'src', 'count']
    train_raw = pd.read_csv(train_file, sep='\t', names=cols, header=None)
    dev_raw = pd.read_csv(dev_file, sep='\t', names=cols, header=None)
    test_raw = pd.read_csv(test_file, sep='\t', names=cols, header=None)
    
    print(f"Loaded records - Train: {len(train_raw)}, Dev: {len(dev_raw)}, Test: {len(test_raw)}")
    
    # 3. Clean datasets
    processed_splits = []
    for df_raw, name in [(train_raw, "train"), (dev_raw, "dev"), (test_raw, "test")]:
        df_clean = pd.DataFrame()
        df_clean['src'] = df_raw['src'].apply(clean_latin_word)
        df_clean['tgt'] = df_raw['tgt'].apply(clean_devanagari_word)
        
        # Drop entries that became empty or have missing pairs
        df_clean = df_clean[(df_clean['src'] != "") & (df_clean['tgt'] != "")].drop_duplicates()
        processed_splits.append(df_clean)
        print(f"Cleaned {name} records: {len(df_clean)}")
        
    train_df, dev_df, test_df = processed_splits
    
    # 4. Build vocabs using clean training set
    src_vocab, tgt_vocab = build_vocabularies(train_df)
    print(f"Vocabularies created - Src (Latin) size: {len(src_vocab)}, Tgt (Devanagari) size: {len(tgt_vocab)}")
    
    # Save vocab.pkl
    vocab_path = os.path.join(data_dir, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump({"src_vocab": src_vocab, "tgt_vocab": tgt_vocab}, f)
    print(f"Saved vocabulary map to '{vocab_path}'")
    
    # 5. Vectorize words to integer sequences
    train_vect = vectorize_dataframe(train_df, src_vocab, tgt_vocab)
    dev_vect = vectorize_dataframe(dev_df, src_vocab, tgt_vocab)
    test_vect = vectorize_dataframe(test_df, src_vocab, tgt_vocab)
    
    # Save vectorized arrays
    with open(os.path.join(data_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_vect, f)
    with open(os.path.join(data_dir, "dev.pkl"), "wb") as f:
        pickle.dump(dev_vect, f)
    with open(os.path.join(data_dir, "test.pkl"), "wb") as f:
        pickle.dump(test_vect, f)
        
    print("Preprocessing completed successfully. Vectorized files written to data/ directory!")

if __name__ == "__main__":
    preprocess_main()
