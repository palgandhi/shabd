# training/merge_and_preprocess.py
"""
Merges the existing Google Dakshina dataset with the AI4Bharat Aksharantar Hindi
dataset (JSONL format), deduplicates, and regenerates all training artifacts:
  data/train.pkl, data/dev.pkl, data/test.pkl, data/vocab.pkl

Usage:
    python3 training/merge_and_preprocess.py \
        --aksharantar-train /tmp/hin_train.json \
        --aksharantar-valid /tmp/hin_valid.json \
        --aksharantar-test  /tmp/hin_test.json

The script is idempotent — safe to re-run.
"""

import os
import re
import pickle
import argparse
import pandas as pd
from tqdm import tqdm


# ── helpers (same logic as preprocess.py) ────────────────────────────────────

def clean_latin(word):
    if not isinstance(word, str):
        return ""
    return re.sub(r'[^a-z]', '', word.lower().strip())

def clean_devanagari(word):
    if not isinstance(word, str):
        return ""
    return "".join(c for c in word.strip() if '\u0900' <= c <= '\u097f')

def build_vocabularies(train_df):
    src_chars, tgt_chars = set(), set()
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building vocab"):
        src_chars.update(list(row['src']))
        tgt_chars.update(list(row['tgt']))
    special = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    src_vocab = {t: i for i, t in enumerate(special)}
    tgt_vocab = {t: i for i, t in enumerate(special)}
    for c in sorted(src_chars):
        if c not in src_vocab:
            src_vocab[c] = len(src_vocab)
    for c in sorted(tgt_chars):
        if c not in tgt_vocab:
            tgt_vocab[c] = len(tgt_vocab)
    return src_vocab, tgt_vocab

def vectorize(df, src_vocab, tgt_vocab):
    sos, eos, unk = 1, 2, 3
    out = []
    for _, row in df.iterrows():
        src = [sos] + [src_vocab.get(c, unk) for c in row['src']] + [eos]
        tgt = [sos] + [tgt_vocab.get(c, unk) for c in row['tgt']] + [eos]
        out.append((src, tgt))
    return out


# ── loaders ───────────────────────────────────────────────────────────────────

def load_dakshina(lexicon_dir="hi/lexicons"):
    """Load the original Dakshina TSV splits."""
    rows = []
    for split in ["train", "dev", "test"]:
        path = os.path.join(lexicon_dir, f"hi.translit.sampled.{split}.tsv")
        if not os.path.exists(path):
            print(f"  [warn] Dakshina file not found: {path} — skipping")
            continue
        df = pd.read_csv(path, sep='\t', names=['tgt', 'src', 'count'], header=None)
        df['split'] = split
        rows.append(df[['src', 'tgt', 'split']])
    if rows:
        df = pd.concat(rows, ignore_index=True)
        print(f"  Dakshina loaded: {len(df):,} raw rows")
        return df
    return pd.DataFrame(columns=['src', 'tgt', 'split'])

def load_aksharantar(train_path, valid_path, test_path):
    """Load Aksharantar JSONL files."""
    import json
    rows = []
    for path, split in [(train_path, 'train'), (valid_path, 'dev'), (test_path, 'test')]:
        if not path or not os.path.exists(path):
            print(f"  [warn] Aksharantar file not found: {path} — skipping")
            continue
        count = 0
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append({
                        'src': obj.get('english word', ''),
                        'tgt': obj.get('native word', ''),
                        'split': split
                    })
                    count += 1
                except json.JSONDecodeError:
                    continue
        print(f"  Aksharantar {split}: {count:,} rows")
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Merge Dakshina + Aksharantar and rebuild .pkl data files")
    parser.add_argument('--aksharantar-train', default='/tmp/hin_train.json')
    parser.add_argument('--aksharantar-valid', default='/tmp/hin_valid.json')
    parser.add_argument('--aksharantar-test',  default='/tmp/hin_test.json')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--lexicon-dir', default='hi/lexicons')
    parser.add_argument('--max-train', type=int, default=None,
                        help='Cap total training pairs (useful for quick experiments)')
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # 1. Load both datasets
    print("\n[1/5] Loading Dakshina...")
    dakshina = load_dakshina(args.lexicon_dir)

    print("\n[2/5] Loading Aksharantar...")
    aksharantar = load_aksharantar(
        args.aksharantar_train,
        args.aksharantar_valid,
        args.aksharantar_test,
    )

    # 2. Combine
    print("\n[3/5] Merging datasets...")
    combined = pd.concat([dakshina, aksharantar], ignore_index=True)

    # Clean
    combined['src'] = combined['src'].apply(clean_latin)
    combined['tgt'] = combined['tgt'].apply(clean_devanagari)
    combined = combined[(combined['src'] != '') & (combined['tgt'] != '')]

    # Deduplicate on (src, tgt) pair — keep first occurrence
    before = len(combined)
    combined = combined.drop_duplicates(subset=['src', 'tgt'])
    print(f"  Deduplication: {before:,} → {len(combined):,} unique pairs")

    # 3. Split
    train_df = combined[combined['split'] == 'train'][['src', 'tgt']].reset_index(drop=True)
    dev_df   = combined[combined['split'] == 'dev'][['src', 'tgt']].reset_index(drop=True)
    test_df  = combined[combined['split'] == 'test'][['src', 'tgt']].reset_index(drop=True)

    if args.max_train and len(train_df) > args.max_train:
        train_df = train_df.sample(args.max_train, random_state=42).reset_index(drop=True)
        print(f"  Capped training set to {args.max_train:,} pairs")

    print(f"\n  Final splits — Train: {len(train_df):,}  Dev: {len(dev_df):,}  Test: {len(test_df):,}")

    # 4. Build vocab from training set only
    print("\n[4/5] Building vocabularies from merged training set...")
    src_vocab, tgt_vocab = build_vocabularies(train_df)
    print(f"  Src vocab: {len(src_vocab)} chars  |  Tgt vocab: {len(tgt_vocab)} chars")

    vocab_path = os.path.join(args.data_dir, 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump({'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}, f)
    print(f"  Saved → {vocab_path}")

    # 5. Vectorize and save
    print("\n[5/5] Vectorizing and saving .pkl files...")
    for split_name, df in [('train', train_df), ('dev', dev_df), ('test', test_df)]:
        vect = vectorize(df, src_vocab, tgt_vocab)
        path = os.path.join(args.data_dir, f'{split_name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(vect, f)
        print(f"  Saved {split_name}.pkl  ({len(vect):,} pairs)")

    print("\n✅ Done! Merged dataset ready. Re-train your models with:")
    print("   python3 training/train_attention.py")
    print("   python3 training/train_seq2seq.py")


if __name__ == '__main__':
    main()
