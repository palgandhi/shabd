# app.py
import os
import pickle
import torch
import random
import numpy as np
from flask import Flask, request, jsonify, render_template

# Import models and inference routines
from models.rule_based import transliterate_word as rule_word, transliterate_sentence as rule_sentence
from models.seq2seq import Seq2Seq, Encoder as Seq2SeqEncoder, Decoder as Seq2SeqDecoder
from models.attention import AttentionSeq2Seq, Encoder as AttentionEncoder, Decoder as AttentionDecoder, Attention
from inference.tokenizer import transliterate_sentence_nn

# Ensure reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

app = Flask(__name__)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

# Global model state
models_state = {
    "vocab": None,
    "seq2seq": None,
    "attention": None,
    "seq2seq_trained": False,
    "attention_trained": False
}

def load_models_at_startup():
    vocab_path = "data/vocab.pkl"
    seq2seq_path = "models/seq2seq_best.pt"
    attention_path = "models/attention_best.pt"
    
    if not os.path.exists(vocab_path):
        print("Startup warning: data/vocab.pkl not found. Neural models cannot be loaded.")
        return
        
    try:
        with open(vocab_path, "rb") as f:
            vocab_data = pickle.load(f)
            models_state["vocab"] = vocab_data
            src_vocab = vocab_data["src_vocab"]
            tgt_vocab = vocab_data["tgt_vocab"]
    except Exception as e:
        print(f"Error loading vocabulary map: {e}")
        return

    # Load Model 2 (Seq2Seq)
    if os.path.exists(seq2seq_path):
        try:
            print("Loading trained standard Seq2Seq weights...")
            encoder2 = Seq2SeqEncoder(len(src_vocab), 64, 256, 512, 0.3)
            decoder2 = Seq2SeqDecoder(len(tgt_vocab), 64, 512, 0.3)
            model2 = Seq2Seq(encoder2, decoder2, device, sos_idx=1).to(device)
            model2.load_state_dict(torch.load(seq2seq_path, map_location=device))
            model2.eval()
            
            models_state["seq2seq"] = model2
            models_state["seq2seq_trained"] = True
            print("Model 2 (Seq2Seq) initialized successfully.")
        except Exception as e:
            print(f"Failed to load Model 2 Seq2Seq: {e}")
    else:
        print("Model 2 (Seq2Seq) checkpoint not found. Starting in rule-based only mode for Model 2.")

    # Load Model 3 (Attention)
    if os.path.exists(attention_path):
        try:
            print("Loading trained Attention Seq2Seq weights...")
            encoder3 = AttentionEncoder(len(src_vocab), 128, 256, 512, 0.4)
            attn = Attention(256, 512, 128)
            decoder3 = AttentionDecoder(len(tgt_vocab), 128, 256, 512, attn, 0.4)
            model3 = AttentionSeq2Seq(encoder3, decoder3, device, sos_idx=1).to(device)
            model3.load_state_dict(torch.load(attention_path, map_location=device))
            model3.eval()
            
            models_state["attention"] = model3
            models_state["attention_trained"] = True
            print("Model 3 (Attention Seq2Seq) initialized successfully.")
        except Exception as e:
            print(f"Failed to load Model 3 Attention Seq2Seq: {e}")
    else:
        print("Model 3 (Attention Seq2Seq) checkpoint not found. Starting in rule-based only mode for Model 3.")

# Run models loading
load_models_at_startup()

@app.route('/')
def index():
    """
    Renders the central comparative sandbox web interface.
    """
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    """
    Accepts Hinglish sentences and transliterates them side-by-side using:
    - Model 1: Rule-Based
    - Model 2: Standard Seq2Seq (LSTM)
    - Model 3: Attention Seq2Seq (LSTM + Attention)
    """
    text = request.json.get('message', '').strip()
    if not text:
        return jsonify({"error": "Empty message inputs provided"}), 400
        
    response_data = {
        "input": text,
        "rule": rule_sentence(text),
        "seq2seq": "[Model 2 Untrained - Baseline Fallback] " + rule_sentence(text),
        "attention": "[Model 3 Untrained - Baseline Fallback] " + rule_sentence(text),
        "seq2seq_trained": models_state["seq2seq_trained"],
        "attention_trained": models_state["attention_trained"]
    }
    
    # Process with standard neural Seq2Seq if trained
    if models_state["seq2seq_trained"]:
        try:
            vocab = models_state["vocab"]
            response_data["seq2seq"] = transliterate_sentence_nn(
                text, models_state["seq2seq"], vocab["src_vocab"], vocab["tgt_vocab"], 
                decode_mode='greedy'
            )
        except Exception as e:
            response_data["seq2seq"] = f"[Model 2 Runtime Error: {e}]"
            
    # Process with Attention neural Seq2Seq if trained
    if models_state["attention_trained"]:
        try:
            vocab = models_state["vocab"]
            response_data["attention"] = transliterate_sentence_nn(
                text, models_state["attention"], vocab["src_vocab"], vocab["tgt_vocab"], 
                decode_mode='beam', beam_width=5
            )
        except Exception as e:
            response_data["attention"] = f"[Model 3 Runtime Error: {e}]"
            
    return jsonify(response_data)

@app.route('/status', methods=['GET'])
def status():
    """
    Returns the current training state of all neural transliteration weights.
    """
    # Reload models on request if they were recently trained
    if not models_state["seq2seq_trained"] or not models_state["attention_trained"]:
        load_models_at_startup()
        
    return jsonify({
        "seq2seq_trained": models_state["seq2seq_trained"],
        "attention_trained": models_state["attention_trained"]
    })

if __name__ == '__main__':
    # Start web app on port 5001
    app.run(debug=True, host='0.0.0.0', port=5001)
