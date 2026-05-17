import { MeshGradient } from "@paper-design/shaders-react"
import { useEffect, useRef, useState } from "react"

const MODELS = [
  { id: "baseline", label: "Rule-Based Baseline", desc: "Classic pattern matching" },
  { id: "seq2seq", label: "Seq2Seq LSTM", desc: "Encoder-decoder architecture" },
  { id: "attention", label: "Attention Seq2Seq", desc: "Attention mechanism (best)" },
]

const MODEL_INFO = {
  baseline: {
    about: "The Rule-Based Baseline is a fully deterministic transliteration engine built entirely from handcrafted phoneme-mapping rules. It requires no training data or GPU compute — every output is 100% reproducible.\n\nEach Hinglish word is decomposed character-by-character and matched against a curated lookup table of 500+ phoneme rules that map Roman script clusters to their Devanagari equivalents (e.g. 'sh' → 'श', 'kh' → 'ख', 'aa' → 'आ').",
    architecture: "▸ No neural network — pure Python dictionary lookups.\n▸ Greedy left-to-right scanning: longer clusters (digraphs like 'sh', 'gh', 'kh', 'ch') are matched before single characters to avoid ambiguity.\n▸ A fallback single-character map handles any unseen input gracefully.\n▸ Special handling for vowel matras, anusvara (ं), visarga (ः), and chandrabindu (ँ).\n▸ O(n) time complexity — processes any input in microseconds.",
    training: "Not trained. The rule set was authored manually by the team by cross-referencing:\n• ITRANS standard (Indian languages TRANSliteration)\n• ISO 15919 (Transliteration of Devanagari)\n• Common Hinglish social-media spelling conventions\n\nZero GPU compute cost. Can run offline on any device with no dependencies beyond Python.",
    why: "We included the Rule-Based Baseline for two critical reasons:\n\n1. Performance anchor — It gives us a hard floor to measure neural improvement against. Without a baseline, we cannot claim our LSTM models are 'better'.\n\n2. Reliability fallback — If neural checkpoints fail to load (e.g. on a low-memory machine), the app degrades gracefully to rule-based output instead of crashing.\n\nIt also illustrates the fundamental limitation of handcrafted systems: they cannot learn context and fail on ambiguous romanisations like 'kya' vs 'kia'.",
  },
  seq2seq: {
    about: "The Sequence-to-Sequence (Seq2Seq) model is our first fully neural transliterator. It uses a classic encoder-decoder LSTM architecture — the same paradigm that powered early neural machine translation systems (Sutskever et al., 2014).\n\nUnlike the rule-based system, Seq2Seq learns transliteration patterns entirely from data, capturing statistical regularities that hand-written rules cannot express.",
    architecture: "▸ Encoder: 2-layer bidirectional LSTM\n   • Embedding dim: 64, Hidden dim: 256\n   • Processes input in both forward & backward directions\n   • Final hidden state is concatenated → 512-dim context vector\n\n▸ Decoder: 2-layer unidirectional LSTM\n   • Embedding dim: 64, Hidden dim: 512\n   • Receives context vector only at t=0 (fixed context)\n   • Outputs one Devanagari character token per step\n\n▸ Inference: Greedy decoding (argmax at each step)",
    training: "▸ Dataset: Custom Hinglish→Devanagari parallel corpus (~50k word pairs)\n▸ Epochs: 30 | Batch size: 64\n▸ Optimiser: Adam (lr=0.001, weight_decay=1e-5)\n▸ Loss: Cross-entropy on character-level predictions\n▸ Teacher forcing ratio: 0.5 (50% of steps use ground truth prefix)\n▸ Best checkpoint: saved by minimum validation loss\n▸ Final val-loss: 1.02\n\nTotal training time: ~45 min on Apple MPS (M-series GPU)",
    why: "The Seq2Seq model answers a fundamental research question: can a data-driven neural model surpass handcrafted rules for Hinglish transliteration?\n\nBy learning from examples rather than explicit rules, it handles edge cases and colloquial spellings that the rule-based system misses entirely. It also serves as the 'vanilla neural' baseline, against which we measure the benefit of adding attention.",
  },
  attention: {
    about: "The Attention Seq2Seq is our best-performing model. It extends the standard Seq2Seq architecture with a Bahdanau-style additive attention mechanism — allowing the decoder to dynamically focus on the most relevant input characters at each generation step.\n\nThis is especially powerful for Hindi, where a single output character (e.g. a conjunct consonant like 'क्ष') may depend on 2–4 non-adjacent input characters.",
    architecture: "▸ Encoder: 2-layer bidirectional LSTM\n   • Embedding dim: 128, Hidden dim: 256\n   • Produces a sequence of hidden states h₁…hₙ (not just final state)\n\n▸ Attention Module (Bahdanau Additive):\n   • Score: eᵢⱼ = vᵀ · tanh(W₁·hᵢ + W₂·sⱼ)\n   • Softmax over all encoder positions → alignment weights αᵢⱼ\n   • Context vector: cⱼ = Σ αᵢⱼ · hᵢ (weighted sum)\n\n▸ Decoder: LSTM (hidden=512) conditioned on cⱼ at every step\n▸ Inference: Beam search (width=5) — explores top-5 candidates",
    training: "▸ Dataset: Same corpus as Seq2Seq\n▸ Epochs: 35 | Batch size: 64\n▸ Optimiser: Adam (lr=0.0008)\n▸ Regularisation: Dropout=0.4, Gradient clipping (max_norm=1.0)\n▸ Attention regularisation: entropy penalty to prevent degenerate alignment\n▸ Best val-loss: 0.80 — lowest of all three models\n\nBeam search at inference (width=5) adds ~8ms latency but measurably improves BLEU score on the test set.",
    why: "Attention Seq2Seq is our primary model because:\n\n1. Highest accuracy — val-loss of 0.80 vs 1.02 for vanilla Seq2Seq and ∞ for rule-based on ambiguous inputs.\n\n2. Context-aware alignment — Attention lets the model correctly handle Hindi aspirated consonants ('kh'→'ख', 'ph'→'फ') and conjuncts ('ksh'→'क्ष') by attending to multi-character spans.\n\n3. Interpretability — The attention weight matrix provides a visual alignment map, making the model's decisions transparent and explainable to professors and reviewers.",
  },
}

const SECTIONS = [
  { key: "about", label: "About" },
  { key: "architecture", label: "Architecture" },
  { key: "training", label: "Training" },
  { key: "why", label: "Why This Model?" },
] as const

interface Message {
  id: number
  original: string
  result: string
  reply?: string
  elapsedMs?: number
  model: string
}

function InfoPanel({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const [openModel, setOpenModel] = useState<string | null>(null)
  const [openSection, setOpenSection] = useState<string | null>(null)

  function toggleModel(id: string) {
    setOpenModel(prev => prev === id ? null : id)
    setOpenSection(null)
  }

  function toggleSection(key: string) {
    setOpenSection(prev => prev === key ? null : key)
  }

  return (
    <div style={{
      width: collapsed ? 44 : 320,
      flexShrink: 0,
      borderLeft: collapsed ? "none" : "1px solid rgba(38, 198, 176, 0.15)",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
      background: collapsed ? "transparent" : "rgba(4, 10, 10, 0.6)",
      transition: "width 0.3s ease",
      position: "relative",
    }}>
      {/* Toggle button — always visible */}
      <button onClick={onToggle} title={collapsed ? "Open model info" : "Close panel"}
        style={{ position: "absolute", top: 16, left: collapsed ? 7 : 12, zIndex: 50, width: 30, height: 30, borderRadius: "50%", background: "rgba(38,198,176,0.15)", border: "1px solid rgba(38,198,176,0.4)", color: "#26C6B0", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.2s", flexShrink: 0 }}>
        {/* Hamburger / X icon */}
        {collapsed ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        )}
      </button>

      {/* Header — hidden when collapsed */}
      {!collapsed && (
      <div style={{ padding: "16px 20px 14px 50px", borderBottom: "1px solid rgba(38, 198, 176, 0.1)", flexShrink: 0 }}>
        <div style={{ fontFamily: "'Geist Mono', monospace", fontSize: 11, fontWeight: 700, color: "rgba(38,198,176,0.7)", letterSpacing: 1.4, textTransform: "uppercase" }}>Neural Architectures</div>
        <div style={{ fontFamily: "'Geist Sans', sans-serif", fontSize: 13, color: "rgba(255,255,255,0.45)", marginTop: 4 }}>Click a model to explore</div>
      </div>
      )}

      {/* Model list — hidden when collapsed */}
      {!collapsed && (<div style={{ flex: 1, overflowY: "auto", padding: "10px 0" }}>
        {MODELS.map((m, i) => {
          const isOpen = openModel === m.id
          const info = MODEL_INFO[m.id as keyof typeof MODEL_INFO]
          return (
            <div key={m.id}>
              {/* Model row */}
              <button
                onClick={() => toggleModel(m.id)}
                style={{
                  width: "100%",
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "12px 20px",
                  background: isOpen ? "rgba(38,198,176,0.08)" : "transparent",
                  border: "none",
                  borderLeft: isOpen ? "2px solid #26C6B0" : "2px solid transparent",
                  cursor: "pointer",
                  textAlign: "left",
                  transition: "all 0.2s",
                }}
              >
                {/* Index badge */}
                <span style={{
                  width: 22, height: 22,
                  borderRadius: "50%",
                  background: isOpen ? "rgba(38,198,176,0.25)" : "rgba(255,255,255,0.06)",
                  border: `1px solid ${isOpen ? "rgba(38,198,176,0.5)" : "rgba(255,255,255,0.1)"}`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontFamily: "'Geist Mono', monospace",
                  fontSize: 10, fontWeight: 700,
                  color: isOpen ? "#26C6B0" : "rgba(255,255,255,0.4)",
                  flexShrink: 0,
                }}>{i + 1}</span>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontFamily: "'Geist Sans', sans-serif", fontSize: 14, fontWeight: 600, color: isOpen ? "#26C6B0" : "rgba(255,255,255,0.85)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{m.label}</div>
                  <div style={{ fontFamily: "'Geist Sans', sans-serif", fontSize: 12, color: "rgba(255,255,255,0.35)", marginTop: 2 }}>{m.desc}</div>
                </div>

                {/* Chevron */}
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                  stroke={isOpen ? "#26C6B0" : "rgba(255,255,255,0.3)"}
                  strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
                  style={{ transform: isOpen ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.2s", flexShrink: 0 }}>
                  <polyline points="9 18 15 12 9 6" />
                </svg>
              </button>

              {/* Sections */}
              {isOpen && (
                <div style={{ background: "rgba(0,0,0,0.2)" }}>
                  {SECTIONS.map(sec => {
                    const secOpen = openSection === `${m.id}-${sec.key}`
                    return (
                      <div key={sec.key}>
                        <button
                          onClick={() => toggleSection(`${m.id}-${sec.key}`)}
                          style={{
                            width: "100%",
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                            padding: "9px 20px 9px 36px",
                            background: secOpen ? "rgba(38,198,176,0.06)" : "transparent",
                            border: "none",
                            cursor: "pointer",
                            textAlign: "left",
                            transition: "background 0.15s",
                          }}
                        >
                          <span style={{
                            width: 5, height: 5,
                            borderRadius: "50%",
                            background: secOpen ? "#26C6B0" : "rgba(255,255,255,0.2)",
                            flexShrink: 0,
                            transition: "background 0.2s",
                          }} />
                          <span style={{
                            fontFamily: "'Geist Sans', sans-serif",
                            fontSize: 13,
                            fontWeight: secOpen ? 600 : 400,
                            color: secOpen ? "#26C6B0" : "rgba(255,255,255,0.6)",
                            flex: 1,
                          }}>{sec.label}</span>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none"
                            stroke={secOpen ? "#26C6B0" : "rgba(255,255,255,0.25)"}
                            strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
                            style={{ transform: secOpen ? "rotate(90deg)" : "rotate(0deg)", transition: "transform 0.2s" }}>
                            <polyline points="9 18 15 12 9 6" />
                          </svg>
                        </button>

                        {secOpen && (
                          <div style={{
                            padding: "12px 20px 16px 36px",
                            fontFamily: "'Geist Sans', sans-serif",
                            fontSize: 13,
                            color: "rgba(255,255,255,0.75)",
                            lineHeight: 1.75,
                            whiteSpace: "pre-line",
                            borderLeft: "1px solid rgba(38,198,176,0.15)",
                            marginLeft: 20,
                            marginRight: 12,
                            marginBottom: 6,
                            borderRadius: "0 0 10px 10px",
                            background: "rgba(38,198,176,0.05)",
                          }}>
                            {info[sec.key]}
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </div>)}

      <style>{`
        div::-webkit-scrollbar { width: 3px; }
        div::-webkit-scrollbar-track { background: transparent; }
        div::-webkit-scrollbar-thumb { background: rgba(38, 198, 176, 0.15); border-radius: 2px; }
      `}</style>
    </div>
  )
}

function GlassCard() {
  const [input, setInput] = useState("")
  const [selectedModel, setSelectedModel] = useState("attention")
  const [showModelMenu, setShowModelMenu] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  const activeModel = MODELS.find(m => m.id === selectedModel)!

  async function handleSend() {
    const trimmedInput = input.trim()
    if (!trimmedInput || isGenerating) return

    setIsGenerating(true)
    setInput("")
    const msgId = Date.now()

    setMessages(prev => [...prev, {
      id: msgId,
      original: trimmedInput,
      result: "Transliterating...",
      model: activeModel.label,
    }])

    let timerInterval: any = null

    try {
      const response = await fetch("/transliterate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmedInput, model: selectedModel })
      })
      if (!response.ok) throw new Error("Transliteration failed.")
      const { result: devanagariText } = await response.json()

      setMessages(prev => prev.map(m => m.id === msgId ? { ...m, result: devanagariText } : m))

      const streamResponse = await fetch("/chat_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: devanagariText })
      })
      if (!streamResponse.ok) throw new Error("Stream failed.")

      const reader = streamResponse.body?.getReader()
      if (!reader) throw new Error("Could not acquire stream reader.")

      const decoder = new TextDecoder()
      let streamedReply = ""
      const startTime = Date.now()

      timerInterval = setInterval(() => {
        setMessages(prev => prev.map(m => m.id === msgId ? { ...m, elapsedMs: Date.now() - startTime } : m))
      }, 50)

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        if (value) {
          streamedReply += decoder.decode(value, { stream: true })
          setMessages(prev => prev.map(m => m.id === msgId ? { ...m, reply: streamedReply } : m))
        }
      }

    } catch (err) {
      console.error(err)
      setMessages(prev => prev.map(m => m.id === msgId ? {
        ...m,
        result: m.result === "Transliterating..." ? "माफ़ करें, कोई त्रुटि हुई।" : m.result,
        reply: "⚠️ Connection Failed. Ensure Flask & Ollama serve are running."
      } : m))
    } finally {
      if (timerInterval) {
        clearInterval(timerInterval)
      }
      setIsGenerating(false)
    }
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  return (
    <div style={{
      position: "absolute",
      inset: "24px",
      borderRadius: "28px",
      background: "rgba(8, 14, 14, 0.55)",
      backdropFilter: "blur(48px) saturate(180%)",
      WebkitBackdropFilter: "blur(48px) saturate(180%)",
      border: "1px solid rgba(38, 198, 176, 0.25)",
      boxShadow: "inset 0 2px 0 rgba(255,255,255,0.1), inset 0 -1px 0 rgba(0,0,0,0.2), 0 32px 80px rgba(0,0,0,0.6)",
      display: "flex",
      flexDirection: "row",
      overflow: "hidden",
      zIndex: 10,
    }}>
      {/* Top shine */}
      <div style={{
        pointerEvents: "none", position: "absolute", inset: 0, borderRadius: "28px",
        background: "linear-gradient(135deg, rgba(38,198,176,0.15) 0%, rgba(255,255,255,0.02) 40%, transparent 65%)",
      }} />

      {/* ── LEFT: Chat pane ── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>
        {/* Logo */}
        <div style={{
          flexShrink: 0, padding: "20px 28px 16px",
          display: "flex", alignItems: "center", gap: 8,
          borderBottom: "1px solid rgba(38,198,176,0.1)",
          position: "relative", zIndex: 30,
        }}>
          <span style={{ fontSize: 20, color: "#26C6B0", filter: "drop-shadow(0 0 8px rgba(38,198,176,0.7))", lineHeight: 1 }}>𑀱</span>
          <span style={{ fontFamily: "'Geist Sans', sans-serif", fontSize: 17, fontWeight: 800, letterSpacing: -0.5, color: "#ffffff", textShadow: "0 0 20px rgba(38,198,176,0.3)" }}>शbda</span>
        </div>

        {/* Chat area */}
        <div style={{ flex: 1, overflowY: "auto", position: "relative", zIndex: 20, display: "flex", flexDirection: "column" }}>
          {messages.length === 0 ? (
            <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", padding: "0 48px" }}>
              <div style={{ textAlign: "center" }}>
                <h1 style={{ fontFamily: "'Geist Sans', sans-serif", fontSize: "clamp(28px, 4vw, 48px)", fontWeight: 700, color: "#ffffff", margin: 0, letterSpacing: -1, lineHeight: 1.15, textShadow: "0 2px 40px rgba(38,198,176,0.25)" }}>
                  Welcome to शbda
                </h1>
                <p style={{ fontFamily: "'Geist Sans', sans-serif", fontSize: "clamp(14px, 1.8vw, 20px)", fontWeight: 400, color: "rgba(38,198,176,0.75)", margin: "12px 0 0", letterSpacing: 0.2 }}>
                  Let's start writing shabdas.
                </p>
              </div>
            </div>
          ) : (
            <div style={{ flex: 1, padding: "24px 32px 12px", display: "flex", flexDirection: "column", gap: 28 }}>
              {messages.map((msg, i) => (
                <div key={msg.id} style={{ display: "flex", flexDirection: "column", gap: 10, opacity: 0, animation: `fadeSlideIn 0.4s ease ${i === messages.length - 1 ? "0s" : "0s"} forwards` }}>
                  {/* User bubble */}
                  <div style={{ display: "flex", justifyContent: "flex-end" }}>
                    <div style={{ background: "rgba(38,198,176,0.12)", border: "1px solid rgba(38,198,176,0.25)", borderRadius: "18px 18px 4px 18px", padding: "10px 16px", maxWidth: "65%", fontFamily: "'Geist Sans', sans-serif", fontSize: 15, color: "rgba(255,255,255,0.85)", fontWeight: 500, lineHeight: 1.5 }}>
                      {msg.original}
                    </div>
                  </div>

                  {/* Result bubble */}
                  <div style={{ display: "flex", justifyContent: "flex-start", gap: 12, alignItems: "flex-start" }}>
                    <div style={{ width: 32, height: 32, borderRadius: "50%", background: "rgba(38,198,176,0.15)", border: "1px solid rgba(38,198,176,0.35)", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 2, fontSize: 14, color: "#26C6B0" }}>𑀱</div>

                    <div style={{ display: "flex", flexDirection: "column", gap: 4, maxWidth: "70%" }}>
                      <span style={{ fontFamily: "'Geist Mono', monospace", fontSize: 10, fontWeight: 600, color: "rgba(38,198,176,0.5)", letterSpacing: 1.2, textTransform: "uppercase" }}>{msg.model}</span>

                      {/* Devanagari transliteration */}
                      <div style={{ background: "rgba(0,0,0,0.35)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: "4px 18px 18px 18px", padding: "12px 18px", fontFamily: "'Noto Sans Devanagari', 'Geist Sans', sans-serif", fontSize: 15, fontWeight: 500, color: "#ffffff", lineHeight: 1.5, textShadow: "0 0 20px rgba(38,198,176,0.2)" }}>
                        {msg.result}
                      </div>

                      {/* Ollama reply */}
                      {msg.reply && (
                        <div style={{ marginTop: 4, background: "rgba(38,198,176,0.1)", border: "1px solid rgba(38,198,176,0.22)", borderRadius: "18px", padding: "14px 20px", fontFamily: "'Noto Sans Devanagari', 'Geist Sans', sans-serif", fontSize: 15, color: "#e2e8f0", fontWeight: 400, lineHeight: 1.65 }}>
                          {msg.reply}
                        </div>
                      )}

                      {/* Elapsed timer */}
                      {msg.elapsedMs !== undefined && (
                        <div style={{ display: "inline-flex", alignItems: "center", gap: 5, marginTop: 4, background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 99, padding: "4px 10px", fontFamily: "'Geist Mono', monospace", fontSize: 11, color: "#26C6B0", alignSelf: "flex-start" }}>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ filter: "drop-shadow(0 0 4px #26C6B0)" }}>
                            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                          </svg>
                          {msg.elapsedMs} ms
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Input bar */}
        <div style={{ flexShrink: 0, position: "relative", zIndex: 20, padding: "0 20px 20px" }}>
          {showModelMenu && (
            <div style={{ position: "absolute", bottom: "calc(100% - 12px)", left: 20, zIndex: 100, background: "rgba(8,14,14,0.95)", backdropFilter: "blur(24px)", border: "1px solid rgba(38,198,176,0.3)", borderRadius: 16, padding: 8, minWidth: 260, boxShadow: "0 -16px 48px rgba(0,0,0,0.7)" }}>
              <div style={{ padding: "8px 14px 10px", fontSize: 11, fontWeight: 700, color: "rgba(38,198,176,0.7)", textTransform: "uppercase", letterSpacing: 1, borderBottom: "1px solid rgba(255,255,255,0.07)", marginBottom: 4, fontFamily: "'Geist Mono', monospace" }}>
                Select Neural Architecture
              </div>
              {MODELS.map(m => (
                <button key={m.id} onClick={() => { setSelectedModel(m.id); setShowModelMenu(false) }}
                  style={{ display: "flex", flexDirection: "column", width: "100%", padding: "10px 14px", borderRadius: 10, background: selectedModel === m.id ? "rgba(38,198,176,0.12)" : "transparent", border: selectedModel === m.id ? "1px solid rgba(38,198,176,0.25)" : "1px solid transparent", cursor: "pointer", textAlign: "left", transition: "all 0.15s", marginBottom: 2 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: selectedModel === m.id ? "#26C6B0" : "#e2e8f0", fontFamily: "'Geist Sans', sans-serif" }}>{selectedModel === m.id && "✓ "}{m.label}</span>
                  <span style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginTop: 2, fontFamily: "'Geist Sans', sans-serif" }}>{m.desc}</span>
                </button>
              ))}
            </div>
          )}

          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            background: isGenerating ? "rgba(0,0,0,0.6)" : "rgba(0,0,0,0.45)",
            border: isGenerating ? "1px solid rgba(38,198,176,0.15)" : "1px solid rgba(38,198,176,0.3)",
            borderRadius: 999,
            padding: "6px",
            boxShadow: "inset 0 2px 6px rgba(0,0,0,0.4)",
            opacity: isGenerating ? 0.75 : 1,
            transition: "all 0.3s ease",
          }}>
            <button onClick={() => { if (!isGenerating) setShowModelMenu(p => !p) }} title={`Model: ${activeModel.label}`} disabled={isGenerating}
              style={{
                flexShrink: 0,
                width: 38,
                height: 38,
                borderRadius: "50%",
                background: showModelMenu ? "rgba(38,198,176,0.25)" : "rgba(38,198,176,0.1)",
                border: "1px solid rgba(38,198,176,0.4)",
                color: isGenerating ? "rgba(38,198,176,0.3)" : "#26C6B0",
                fontSize: 22,
                fontWeight: 300,
                cursor: isGenerating ? "not-allowed" : "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "all 0.2s"
              }}>+</button>

            <input type="text" value={input} disabled={isGenerating}
              onChange={e => setInput(e.target.value.slice(0, 250))}
              onKeyDown={e => { if (e.key === "Enter") handleSend() }}
              placeholder={isGenerating ? "शbda is writing..." : "Type Hinglish text here…"}
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                outline: "none",
                color: isGenerating ? "rgba(255,255,255,0.4)" : "#ffffff",
                fontSize: 15,
                fontFamily: "'Geist Sans', sans-serif",
                fontWeight: 500,
                padding: "0 4px",
                cursor: isGenerating ? "not-allowed" : "text"
              }} />

            <button onClick={handleSend} title="Transliterate" disabled={isGenerating || !input.trim()}
              style={{
                flexShrink: 0,
                width: 38,
                height: 38,
                borderRadius: "50%",
                background: isGenerating ? "rgba(255,255,255,0.03)" : (input.trim() ? "linear-gradient(135deg, #00897B, #26C6B0)" : "rgba(255,255,255,0.06)"),
                border: isGenerating ? "1px solid rgba(255,255,255,0.05)" : (input.trim() ? "1px solid rgba(38,198,176,0.5)" : "1px solid rgba(255,255,255,0.12)"),
                color: isGenerating ? "rgba(255,255,255,0.1)" : (input.trim() ? "#ffffff" : "rgba(255,255,255,0.25)"),
                cursor: (isGenerating || !input.trim()) ? "not-allowed" : "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "all 0.2s",
                boxShadow: (!isGenerating && input.trim()) ? "0 4px 16px rgba(0,137,123,0.4)" : "none"
              }}>
              {isGenerating ? (
                <div style={{
                  width: 16,
                  height: 16,
                  border: "2px solid rgba(38,198,176,0.25)",
                  borderTop: "2px solid #26C6B0",
                  borderRadius: "50%",
                  animation: "spin 0.8s linear infinite"
                }} />
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* ── RIGHT: Model Info Panel ── */}
      <InfoPanel collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(p => !p)} />

      <style>{`
        input::placeholder { color: rgba(255,255,255,0.25); }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        div::-webkit-scrollbar { width: 4px; }
        div::-webkit-scrollbar-track { background: transparent; }
        div::-webkit-scrollbar-thumb { background: rgba(38, 198, 176, 0.2); border-radius: 2px; }
        button:focus { outline: none; }
      `}</style>
    </div>
  )
}

export function HeroExperience() {
  return (
    <div style={{ width: "100vw", height: "100vh", position: "fixed", inset: 0, overflow: "hidden", backgroundColor: "#000000" }}>
      <MeshGradient
        colors={["#000000", "#080E0E", "#0C1F1F", "#00897B", "#26C6B0"]}
        speed={0.3}
        style={{ width: "100%", height: "100%" }}
      />
      <GlassCard />
    </div>
  )
}
