# शbda (शbda) - Hybrid Hinglish-to-Hindi Transliteration & Conversational AI Dashboard

शbda is a state-of-the-art, hybrid Roman-to-Devanagari (Hinglish-to-Hindi) phonetic transliteration engine coupled with a local conversational AI assistant. It integrates classical linguistic mapping rules, custom-trained deep neural networks (LSTMs & Attention networks), and local Large Language Models (Ollama Gemma) into a premium, responsive glassmorphism workspace.

---

## 🚀 Key Features

*   **Hybrid Transliteration Engine**: Supports three distinct architectural variants:
    1.  **Rule-Based Baseline (Classical)**: Deterministic, character-level phonetic mapping with advanced Hinglish heuristic corrections (e.g., trailing vowels, character clusters).
    2.  **Standard Seq2Seq (Neural)**: A PyTorch-trained character-level Sequence-to-Sequence (LSTM) Encoder-Decoder model.
    3.  **Seq2Seq with Bahdanau Attention (Neural + Search)**: Custom PyTorch model utilizing active attention decoding paired with Beam Search (`beam_width=5`) for state-of-the-art accuracy.
*   **Devanagari Punctuation-Aware**: Intelligently intercepts English full stops (`.`) and automatically converts them to Devanagari dandas (`।`) across all models.
*   **Friendly Hindi Conversational AI**: Integrates locally with Ollama running `gemma4:e2b` to generate conversational replies streamed token-by-token directly to the chat interface.
*   **Concurrency & Stream Locking**: Programmatic UI locks (`isGenerating` state) that disable inputs, block duplicate submissions, and display custom spinner states during inference.
*   **State-of-the-Art Glassmorphic UI**: Beautiful responsive layout containing:
    *   Dynamic translucent panels, real-time message bubble streams, and micro-interactions.
    *   An elegant collapsible sidebar detailing training metrics, dataset choices, and architectural rationales for each transliteration model.

---

## 📂 Project Architecture

```
.
├── app.py                      # Core Flask API (routing, stream generation)
├── ollama_client.py            # Local Ollama client (stream API & history limits)
├── requirements.txt            # Python dependencies
├── models/                     
│   ├── rule_based.py           # Phonetic heuristic lookup engine
│   ├── seq2seq.py              # PyTorch Encoder-Decoder LSTM
│   └── attention.py            # PyTorch Seq2Seq + Bahdanau Attention Decoder
├── inference/                  
│   ├── tokenizer.py            # Character-level sentence parser
│   └── beam_search.py          # Greedy and Beam Search decoding algorithms
├── training/                   
│   └── preprocess.py           # Corpus cleaning, tokenization, & dataset building
└── frontend-react/             
    ├── src/                    
    │   ├── components/         
    │   │   └── HeroExperience.tsx  # Core high-fidelity Dashboard component
    │   └── index.css           # Global custom typography and aesthetics
    ├── package.json            # Node project configuration
    └── vite.config.ts          # Vite build configurations
```

---

## ⚡ Quick Start

### 1. Prerequisite Checklist

Ensure you have the following installed on your system:
*   Python 3.10+
*   Node.js v18+ & npm
*   [Ollama Desktop](https://ollama.com/)

### 2. Setup the Python Backend

From the repository root:

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull and run the local Ollama Gemma Model
ollama run gemma4:e2b

# 4. Start the Flask api backend
python3 app.py
```
The Flask backend will start on `http://127.0.0.1:5001`.

### 3. Setup the React Frontend

From the `frontend-react` folder:

```bash
# 1. Navigate to the frontend directory
cd frontend-react

# 2. Install dependencies
npm install

# 3. Run the development server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🧠 Transliteration Models Summary

| Model ID | Architecture | Features / Strengths | Decoding Strategy |
| :--- | :--- | :--- | :--- |
| **Model 1: Baseline** | Rule-Based (Heuristics) | Deterministic mapping, zero VRAM, immediate execution | Character Mapping Tables |
| **Model 2: Seq2Seq** | Encoder-Decoder LSTM | Learns context-free phonetic mappings from parallel corpus | Greedy Search |
| **Model 3: Attention** | Bahdanau Attention LSTM | Dynamically tracks target character contexts during generation | Beam Search (`beam_width=5`) |

---

## 🛡️ License

This project is licensed under the MIT License - see the LICENSE file for details.
