# Hinglish Chatbot — Project Plan v2

**English-to-Hindi Transliterator + Conversational Layer**
Dataset: Dakshina (Google Research) · Stack: PyTorch + Flask + Ollama

---

## 0. Read This First — Decision Point on Ollama

This plan assumes **Ollama is allowed** because the only exam rule is *"the core transliteration model must be trained from scratch."* Ollama is a downstream service that generates conversational replies; it is *not* our submitted/graded model.

**Before writing a single line of code, confirm this with the examiner.** Two outcomes:

- **Ollama allowed** → follow this plan as-is.
- **Ollama not allowed** → drop Section 5, reposition the project as a "Hinglish-to-Hindi Transliteration System with Web Interface", and skip the `/chat` endpoint (keep only `/transliterate` and `/compare`). Everything else stays the same.

Either way, the *core deliverable* (the trained transliteration model) is identical, so we lose nothing by starting.

---

## 1. Project Overview

### 1.1 What we are building

A web chat interface where:

1. The user types in **Hinglish** (Hindi typed phonetically in English letters).
2. **Our custom-trained transliteration model** converts the script to Devanagari.
3. **Ollama** (running an open-source LLM) reads the Devanagari text and generates a conversational Hindi reply.
4. The full conversation is shown in the UI.

### 1.2 What the trained model does and does not do

- **It does:** convert Latin-script Hindi to Devanagari script. Pure character-level mapping.
- **It does not:** understand the question, generate replies, translate English to Hindi, or do any semantics.

This distinction is important when explaining the project. Our contribution is the **transliterator**. The chatbot wrapper is a demo container.

### 1.3 Mapping to exam guidelines

| Guideline | How it is satisfied |
|---|---|
| Hosted with user-friendly chat interface | Flask app + HTML/CSS/JS chat UI |
| Multiple models trained and compared | Three transliteration models with side-by-side comparison endpoint |
| Core model trained from scratch | All weights randomly initialized; trained only on Dakshina; checkpoints + loss curves prove this |

### 1.4 Pipeline (example)

```
User input:    "Ye pustak ka naam kya hai?"
                ↓
Tokenize:      ["Ye", "pustak", "ka", "naam", "kya", "hai"]
                ↓
Per-word transliterate (our model):
               ["ये", "पुस्तक", "का", "नाम", "क्या", "है"]
                ↓
Rejoin + punct: "ये पुस्तक का नाम क्या है?"
                ↓
Send to Ollama (with system prompt) → Hindi reply
                ↓
UI shows: Hinglish input, Hindi transliteration, Hindi reply
```

---

## 2. Dataset

### 2.1 Source

**Dakshina** by Google Research — `github.com/google-research-datasets/dakshina`.

We use only the **Hindi lexicon** (`dakshina_dataset_v1.0/hi/lexicons/`):

- `hi.translit.sampled.train.tsv` (~25K–30K word pairs — verify exact count after download)
- `hi.translit.sampled.dev.tsv`
- `hi.translit.sampled.test.tsv`

Format per line: `devanagari_word \t romanized_word \t count`

### 2.2 Preprocessing pipeline (Phase 1 deliverable)

```
1. Load all three TSVs
2. Drop rows with count column problems, nulls, duplicates
3. Lowercase Latin input (preserve Devanagari as-is)
4. Filter: keep only words with chars in [a-z] and [Devanagari Unicode block]
5. Build two character vocabularies:
   - src_vocab: Latin chars + <SOS>, <EOS>, <PAD>, <UNK>
   - tgt_vocab: Devanagari chars + matras + halant + <SOS>, <EOS>, <PAD>, <UNK>
6. Convert chars → integer indices
7. Save: vocab.pkl, train.pkl, dev.pkl, test.pkl
```

### 2.3 Sentence-level handling (for the chatbot)

The model is trained word-by-word, but the user types sentences. At inference:

```python
def transliterate_sentence(text):
    # Split on whitespace, preserve punctuation positions
    tokens = re.findall(r"[a-zA-Z]+|[^\s\w]+|\s+|\d+", text)
    out = []
    for tok in tokens:
        if re.fullmatch(r"[a-zA-Z]+", tok):
            out.append(model.transliterate(tok.lower()))
        else:
            out.append(tok)   # punctuation, digits, spaces pass through
    return "".join(out)
```

Handles: punctuation, numbers, mixed English words (passes them through unchanged — acceptable for a demo).

---

## 3. Models

We build **three models** as required. All are trained from scratch — no pretrained weights, no fine-tuning of existing models.

### 3.1 Comparison overview

| # | Model | Complexity | Realistic Word Accuracy | Role |
|---|---|---|---|---|
| 1 | Rule-Based (proper Devanagari logic) | Low | ~55–65% | Baseline |
| 2 | LSTM Seq2Seq (no attention) | Medium | ~70–78% | Mid-tier trained model |
| 3 | LSTM Seq2Seq + Bahdanau Attention | High | ~82–90% | **Primary model** ⭐ |
| 4 (optional) | Mini Transformer (2-layer) | High | ~85–92% | Stretch goal |

Numbers are realistic-honest, not aspirational.

### 3.2 Model 1 — Rule-Based Transliterator (rewritten properly)

The version in the original plan would fail on real Hindi because it treats every Latin character as mapping to a *standalone* Devanagari letter. Real Devanagari uses **consonant + vowel matra** combinations and **halant (्)** for consonant clusters. Without these, "namaste" becomes garbage instead of **नमस्ते**.

Proper structure:

```python
# models/rule_based.py
CONSONANTS = {
    'kh':'ख','gh':'घ','ch':'च','chh':'छ','jh':'झ','th':'थ',
    'dh':'ध','ph':'फ','bh':'भ','sh':'श','zh':'झ','ng':'ङ',
    'k':'क','g':'ग','j':'ज','t':'त','d':'द','n':'न',
    'p':'प','b':'ब','m':'म','y':'य','r':'र','l':'ल',
    'v':'व','w':'व','s':'स','h':'ह','z':'ज़','f':'फ़',
}
VOWELS_STANDALONE = {
    'aa':'आ','ee':'ई','oo':'ऊ','ai':'ऐ','au':'औ',
    'a':'अ','i':'इ','u':'उ','e':'ए','o':'ओ',
}
VOWEL_MATRAS = {
    'aa':'ा','ee':'ी','oo':'ू','ai':'ै','au':'ौ',
    'a':'','i':'ि','u':'ु','e':'े','o':'ो',   # 'a' = inherent vowel
}
HALANT = '्'

def transliterate_word(word):
    word = word.lower()
    result, i = '', 0
    while i < len(word):
        # Try longest consonant match (3 chars then 2 then 1)
        cons = next((c for L in (3,2,1) if (c := word[i:i+L]) in CONSONANTS), None)
        if cons:
            result += CONSONANTS[cons]
            i += len(cons)
            # Look for following vowel → use matra
            vowel = next((v for L in (2,1) if (v := word[i:i+L]) in VOWEL_MATRAS), None)
            if vowel:
                result += VOWEL_MATRAS[vowel]
                i += len(vowel)
            elif i < len(word) and word[i] in 'bcdfghjklmnpqrstvwxyz':
                result += HALANT   # consonant cluster
        else:
            # Standalone vowel (start of word, or after another vowel)
            vowel = next((v for L in (2,1) if (v := word[i:i+L]) in VOWELS_STANDALONE), None)
            if vowel:
                result += VOWELS_STANDALONE[vowel]
                i += len(vowel)
            else:
                i += 1   # unknown char, skip
    return result
```

This will actually give you ~55–65% word accuracy on Dakshina, which is a legitimate baseline. Good baseline beats a broken baseline.

### 3.3 Model 2 — LSTM Seq2Seq

**Architecture:**
- Encoder: 1-layer Bidirectional LSTM (256 hidden units each direction)
- Decoder: 1-layer LSTM (512 hidden units)
- Embeddings: 64-dim, separate for src and tgt vocab
- Dropout: 0.3

**Training config:**
```python
EMBEDDING_DIM = 64
HIDDEN_DIM    = 256
EPOCHS        = 30
BATCH_SIZE    = 64
LR            = 0.001
DROPOUT       = 0.3
TEACHER_FORCE = 0.5
SEED          = 42        # reproducibility
```

**Loss:** CrossEntropyLoss (ignore_index = PAD)
**Optimizer:** Adam with `lr=0.001`, β=(0.9, 0.98)
**LR schedule:** `ReduceLROnPlateau` on validation CER

**Expected training time:** ~25–35 min on Colab T4 for 30 epochs.

### 3.4 Model 3 — LSTM Seq2Seq + Bahdanau Attention ⭐

**Architecture:**
- Encoder: 2-layer Bidirectional LSTM (256 each direction)
- **Bahdanau additive attention** over all encoder outputs
- Decoder: 2-layer LSTM (512 hidden), attends to encoder at every step
- Embeddings: 128-dim
- Dropout: 0.4

**Training config:**
```python
EMBEDDING_DIM = 128
HIDDEN_DIM    = 512
EPOCHS        = 50
BATCH_SIZE    = 64
LR            = 0.0005
DROPOUT       = 0.4
TEACHER_FORCE = 0.5         # anneal to 0.3 after epoch 30
GRAD_CLIP     = 1.0
SEED          = 42
```

**Save checkpoint every 5 epochs.** Keep best-by-validation-CER as `attention_best.pt`.

**Inference:** **Beam search with width 5** (not greedy). This typically adds 5–10% word accuracy with no retraining — easy win.

**Expected training time:** ~70–90 min on Colab T4 for 50 epochs.

### 3.5 (Optional) Model 4 — Mini Transformer

If a member finishes early. 2 encoder + 2 decoder layers, 4 heads, d_model=128. Often beats LSTM+Attention on character-level tasks and trains faster. Skip if time-constrained.

### 3.6 Evaluation Metrics (run on test split for all models)

| Metric | What it measures | Target (Model 3) |
|---|---|---|
| **Word Accuracy (Top-1)** | % of words transliterated exactly correctly | > 82% |
| **Top-5 Accuracy** | % where correct answer is in top-5 beam | > 93% |
| **Character Error Rate (CER)** | Edit distance / target length | < 8% |
| **Mean Edit Distance** | Avg Levenshtein per word | < 0.5 |

Dropped BLEU — it's a translation metric, not appropriate for character-level transliteration.

### 3.7 Required training artifacts ("from scratch" proof)

For the examiner to see this was actually trained from scratch:

1. **Loss curves** — `training/loss_curves.png` showing train + validation loss per epoch for Models 2 and 3.
2. **Checkpoint files** with epoch numbers in filenames.
3. **Training log** — text file showing initial loss is high (~4–5) and drops, proving random initialization.
4. **`model.summary()` printouts** showing parameter count + that all params are trainable.

Save these to `training/artifacts/`. Show them in the demo or the README.

---

## 4. System Architecture

### 4.1 Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    FLASK WEB SERVER                        │
│                                                            │
│  Browser  ──POST /chat──►  app.py                         │
│                              │                             │
│                ┌─────────────▼─────────────┐               │
│                │  Sentence tokenizer       │               │
│                │  + per-word transliterator│               │
│                │  (LSTM + Attention, beam) │               │
│                └─────────────┬─────────────┘               │
│                              │                             │
│                ┌─────────────▼─────────────┐               │
│                │  Ollama (local)            │              │
│                │  qwen2.5:7b / aya-expanse  │              │
│                │  + system prompt + history │              │
│                └─────────────┬─────────────┘               │
│                              │                             │
│  Browser  ◄──JSON response───┘                            │
└────────────────────────────────────────────────────────────┘
```

### 4.2 Folder structure

```
project/
├── training/
│   ├── 01_preprocess.ipynb
│   ├── 02_train_seq2seq.ipynb
│   ├── 03_train_attention.ipynb
│   ├── 04_evaluate.ipynb              # runs all metrics on test split
│   └── artifacts/
│       ├── loss_curves.png
│       └── training_log.txt
├── models/
│   ├── rule_based.py
│   ├── seq2seq.py
│   ├── attention.py
│   ├── vocab.pkl
│   ├── seq2seq_best.pt
│   └── attention_best.pt
├── inference/
│   ├── tokenize.py                    # sentence → tokens
│   └── beam_search.py
├── app.py                             # Flask backend
├── ollama_client.py                   # wraps Ollama calls + history
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── tests/
│   └── test_examples.py               # 20 hand-picked sentences
├── README.md
└── requirements.txt
```

### 4.3 Tech stack (pinned versions)

| Layer | Tech | Version |
|---|---|---|
| Language | Python | 3.10 |
| ML framework | PyTorch | 2.1.0 |
| Backend | Flask | 3.0.0 |
| LLM runtime | Ollama | latest (≥ 0.3) |
| LLM model | `qwen2.5:7b` or `aya-expanse:8b` | — |
| Training | Google Colab T4 | — |
| HTTP | requests | 2.31.0 |
| Tokenizing | (stdlib `re`) | — |
| Plotting | matplotlib | 3.8.0 |

---

## 5. Ollama Integration

### 5.1 Model choice — **NOT** Llama 3

Llama 3's Hindi is unreliable — frequently replies in English or broken Hindi. Test these in order and pick the best:

1. `qwen2.5:7b` — strong Hindi, fast on consumer GPU.
2. `aya-expanse:8b` — Cohere's multilingual model, designed for low-resource languages including Hindi.
3. `gemma2:9b` — fallback.

Pull during setup: `ollama pull qwen2.5:7b`

### 5.2 System prompt (critical — without this, replies are bad)

```python
SYSTEM_PROMPT = """आप एक मित्रवत हिंदी सहायक हैं।
उपयोगकर्ता आपसे हिंदी में बात करेगा।
हमेशा हिंदी (देवनागरी लिपि) में उत्तर दें।
उत्तर संक्षिप्त रखें — अधिकतम 3 वाक्य।
यदि प्रश्न स्पष्ट नहीं है, तो विनम्रता से और जानकारी मांगें।
अंग्रेजी का प्रयोग न करें।"""
```

Translation for the README: "You are a friendly Hindi assistant. The user will speak to you in Hindi. Always reply in Hindi (Devanagari script). Keep replies short — max 3 sentences. If the question is unclear, politely ask for more info. Do not use English."

### 5.3 `ollama_client.py` (with conversation history)

```python
import requests

class OllamaChat:
    def __init__(self, model="qwen2.5:7b", system_prompt=SYSTEM_PROMPT, max_history=6):
        self.model = model
        self.history = [{"role": "system", "content": system_prompt}]
        self.max_history = max_history

    def chat(self, user_hindi):
        self.history.append({"role": "user", "content": user_hindi})
        try:
            r = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": self.history,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 256}
                },
                timeout=30
            )
            r.raise_for_status()
            reply = r.json()["message"]["content"]
            self.history.append({"role": "assistant", "content": reply})
            # keep history bounded: system + last N exchanges
            if len(self.history) > 1 + 2 * self.max_history:
                self.history = [self.history[0]] + self.history[-2*self.max_history:]
            return reply
        except requests.RequestException as e:
            return f"[Ollama error — is `ollama serve` running? {e}]"
```

Uses the `/api/chat` endpoint (proper multi-turn) instead of `/api/generate` (one-shot).

---

## 6. Flask Application

### 6.1 Endpoints

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/` | GET | — | Renders chat UI |
| `/chat` | POST | `{"message": "..."}` | `{"hindi_input": "...", "reply": "..."}` |
| `/transliterate` | POST | `{"message": "..."}` | `{"hindi": "..."}` (no LLM call) |
| `/compare` | POST | `{"message": "..."}` | `{"rule": "...", "seq2seq": "...", "attention": "..."}` |
| `/reset` | POST | — | Clears conversation history |

### 6.2 `app.py` skeleton

```python
from flask import Flask, request, jsonify, render_template
import torch, random, numpy as np
from models.attention import AttentionSeq2Seq
from models.seq2seq import Seq2Seq
from models.rule_based import transliterate_word as rule_translit
from inference.tokenize import transliterate_sentence
from ollama_client import OllamaChat

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

app = Flask(__name__)

# Load all three models once at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attention_model = AttentionSeq2Seq(...).to(device)
attention_model.load_state_dict(torch.load('models/attention_best.pt', map_location=device))
attention_model.eval()

seq2seq_model = Seq2Seq(...).to(device)
seq2seq_model.load_state_dict(torch.load('models/seq2seq_best.pt', map_location=device))
seq2seq_model.eval()

ollama = OllamaChat(model="qwen2.5:7b")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    text = request.json.get('message', '').strip()
    if not text:
        return jsonify({"error": "Empty message"}), 400
    try:
        hindi = transliterate_sentence(text, attention_model, beam_width=5)
        reply = ollama.chat(hindi)
        return jsonify({"hindi_input": hindi, "reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare():
    text = request.json.get('message', '').strip()
    return jsonify({
        "rule":      transliterate_sentence(text, rule_translit),
        "seq2seq":   transliterate_sentence(text, seq2seq_model, beam_width=1),
        "attention": transliterate_sentence(text, attention_model, beam_width=5),
    })

@app.route('/reset', methods=['POST'])
def reset():
    ollama.history = ollama.history[:1]   # keep system prompt only
    return jsonify({"ok": True})

if __name__ == '__main__':
    app.run(debug=False, port=5000)
```

### 6.3 Frontend requirements

- WhatsApp-style chat bubbles (user right, AI left).
- Each user message displays: original Hinglish (small grey) + transliterated Hindi (main).
- **Loading state** while waiting for response (typing dots).
- **Error state** if Ollama is down — show "AI service unavailable, transliteration still works."
- **"Compare Models" button** opens a modal showing all 3 outputs side-by-side.
- **"Reset Conversation" button** clears history.
- Mobile-responsive (test on phone width).

---

## 7. Development Timeline — **24 Hours, Parallel Tracks**

The original plan was 7 hours of sequential training (Phases 3 and 4 back-to-back on the critical path). With three people, we can run two training jobs concurrently and develop the frontend with a stub model.

### 7.1 Critical path = ~14 hours (not 24)

```
Hour 0 ──── Phase 1: All hands on preprocessing (1.5 hr)
Hour 1.5 ── Three parallel tracks begin
            ├── Member 1: Flask + Ollama setup (uses rule-based as stub)
            ├── Member 2: Model 2 training (Colab account A)
            └── Member 3: Model 3 training (Colab account B) + rule-based model
Hour 4   ── Member 2 done. Helps Member 1 on frontend.
Hour 6   ── Frontend done. Member 1 + 2 start integration.
Hour 9   ── Model 3 done. Plug into Flask, replace stub.
Hour 10  ── Full integration test.
Hour 11  ── Evaluation script run, metrics computed.
Hour 12  ── Bug fixes, demo prep.
Hour 14  ── Submission-ready.
Hour 14–24 Buffer / polish / sleep
```

### 7.2 Phase breakdown

| Phase | Task | Owner | Hours | Deliverable |
|---|---|---|---|---|
| 1 | Dataset download, preprocessing, vocab | All 3 | 1.5 | `vocab.pkl`, train/dev/test `.pkl` |
| 2a | Rule-based model (proper Devanagari) | M3 | 1.5 | `rule_based.py`, tested |
| 2b | LSTM Seq2Seq training | M2 (Colab A) | 3 | `seq2seq_best.pt` + loss curve |
| 2c | LSTM+Attention training | M3 (Colab B) | 6 | `attention_best.pt` + loss curve |
| 3 | Beam search inference module | M2 | 1 | `beam_search.py` |
| 4 | Ollama setup + client | M1 | 1 | `ollama_client.py`, system prompt tuned |
| 5 | Flask backend (with stub model) | M1 | 2 | `app.py`, all endpoints |
| 6 | Frontend (chat UI) | M1 + M2 | 2.5 | `index.html`, `script.js`, `style.css` |
| 7 | Integration + swap stub for real model | All 3 | 1.5 | End-to-end working demo |
| 8 | Evaluation script + metrics | M3 | 1 | `04_evaluate.ipynb` results |
| 9 | Demo prep, README, polish | All 3 | 2 | Final submission |

### 7.3 Team responsibilities

- **Member 1 (Backend/UI):** Flask app, Ollama integration, frontend chat UI, integration testing.
- **Member 2 (Mid-tier ML):** Model 2 training, beam search code, evaluation script, frontend assist.
- **Member 3 (Lead ML):** Rule-based model (proper version), Model 3 training, evaluation metrics, training artifacts.

---

## 8. Setup & Installation

### 8.1 `requirements.txt` (pinned)

```
torch==2.1.0
flask==3.0.0
requests==2.31.0
numpy==1.26.0
pandas==2.1.0
scikit-learn==1.3.0
matplotlib==3.8.0
tqdm==4.66.0
python-Levenshtein==0.23.0
```

### 8.2 Install & run

```bash
# 1. Clone the project
cd project/

# 2. Python dependencies
pip install -r requirements.txt

# 3. Install Ollama from https://ollama.com/download
# 4. Pull the Hindi-capable model
ollama pull qwen2.5:7b

# 5. Start Ollama (separate terminal)
ollama serve

# 6. Run Flask
python app.py

# 7. Open browser → http://localhost:5000
```

### 8.3 Training (Colab)

For each `.ipynb`:
1. Open notebook → Runtime → Change runtime type → **T4 GPU**.
2. Run all cells.
3. Download the resulting `.pt` file to local `models/`.

---

## 9. Demo Strategy

### 9.1 Pick demo inputs that showcase the pipeline

Avoid ambiguous prompts ("Ye pustak ka naam kya hai?" with no context will get a vague reply, regardless of how good your model is). Use prompts that produce strong, demonstrably-Hindi replies:

- `"Aaj mausam kaisa hai?"` — weather chitchat
- `"Mujhe ek chhota joke sunao"` — joke request, fluent reply
- `"Bharat ki rajdhani kya hai?"` — factual, confident reply
- `"Tum kaun ho?"` — chatbot self-intro, shows persona
- `"Main udaas hoon"` — emotional, shows empathetic reply
- `"Ek se das tak ginti karo"` — counting, shows numeric Hindi

### 9.2 Demo script (5 minutes)

1. (30s) Architecture diagram on screen, explain the split: "our model does the script, off-the-shelf LLM does the reply."
2. (30s) Show training loss curve. Highlight initial loss → final loss to prove "from scratch."
3. (1 min) Live chat with 3–4 inputs from the list above.
4. (1 min) Click "Compare Models" — show all 3 outputs side-by-side for a word the rule-based model gets wrong (e.g. "knowledge" → both seq2seq models get it, rule-based fails).
5. (1 min) Show evaluation metrics table (Word Acc, Top-5, CER, Edit Distance for all 3 models).
6. (1 min) Q&A buffer.

---

## 10. Submission Checklist

| Item | Done | Notes |
|---|---|---|
| Rule-based model (proper Devanagari logic) | ☐ | `rule_based.py` |
| LSTM Seq2Seq trained | ☐ | `seq2seq_best.pt` |
| LSTM+Attention trained | ☐ | `attention_best.pt` |
| Beam search inference | ☐ | `beam_search.py` |
| Training loss curves saved | ☐ | `training/artifacts/loss_curves.png` |
| Training log (proves from-scratch) | ☐ | `training/artifacts/training_log.txt` |
| Evaluation metrics computed | ☐ | All 4 metrics × 3 models in README |
| Flask app running on `localhost:5000` | ☐ | — |
| Ollama integrated (non-Llama-3 model) | ☐ | `qwen2.5:7b` or `aya-expanse:8b` |
| System prompt tuned | ☐ | Hindi-only enforced |
| Conversation history (`/api/chat`) | ☐ | — |
| Sentence-level transliteration | ☐ | Handles punct, numbers, spaces |
| Chat UI with loading + error states | ☐ | — |
| Compare-Models feature | ☐ | `/compare` endpoint + UI button |
| Reset Conversation button | ☐ | `/reset` endpoint |
| Mobile responsive | ☐ | Test at 375px width |
| README with metrics + dataset citation | ☐ | Cite Dakshina (Google Research) |
| End-to-end tested with 10+ inputs | ☐ | — |

---

## 11. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Ollama not allowed | Medium | Drop Section 5, ship transliterator-only (see Section 0) |
| Attention model doesn't converge in 50 epochs | Low | Reduce LR, train 20 more epochs, or fall back to Model 2 as primary |
| Colab disconnect during training | High | Save checkpoint every 5 epochs; resume from last checkpoint |
| Ollama replies in English | High (with Llama 3) / Low (with Qwen/Aya) | Use Qwen 2.5 or Aya; strict system prompt |
| Out-of-vocab Hinglish words at demo | Medium | Beam search handles novel chars well; fall back to rule-based for unknown chars |
| Hindi text rendering broken in browser | Low | Set `<meta charset="utf-8">`, use `font-family: 'Noto Sans Devanagari', sans-serif;` |
| Last-minute integration breaks | Medium | Member 1 develops Flask against rule-based stub from hour 1.5; swap is 1-line change |

---

## 12. Things We Are Explicitly Not Doing (and why)

Keep scope tight. Do not be tempted to add:

- **Voice input / TTS** — fun but eats 3+ hours and is irrelevant to graded criteria.
- **English-to-Hindi translation** — outside scope; we transliterate, we don't translate.
- **Fine-tuning Ollama's LLM** — violates "no pretrained" if interpreted strictly, and we don't have time.
- **A 4th custom intent classifier** — Ollama handles intent implicitly.
- **Reverse direction (Hindi → Hinglish)** — not asked for.

---

**Good luck. Build the rule-based model first (1.5 hours, low risk), start both training jobs by hour 1.5, and you have 12 hours of buffer.**
