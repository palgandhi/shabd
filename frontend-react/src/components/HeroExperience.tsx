import { MeshGradient } from "@paper-design/shaders-react"
import { useEffect, useRef, useState } from "react"

const MODELS = [
  { id: "baseline", label: "Rule-Based Baseline", desc: "Classic pattern matching" },
  { id: "seq2seq", label: "Seq2Seq LSTM", desc: "Encoder-decoder architecture" },
  { id: "attention", label: "Attention Seq2Seq", desc: "Attention mechanism (best)" },
]

interface Message {
  id: number
  original: string
  result: string
  model: string
}

// Mock transliterator — swap for real API call
function mockTransliterate(text: string): string {
  return `[ ${text} ] → देवनागरी`
}

function GlassCard() {
  const [input, setInput] = useState("")
  const [selectedModel, setSelectedModel] = useState("attention")
  const [showModelMenu, setShowModelMenu] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)

  const activeModel = MODELS.find(m => m.id === selectedModel)!

  function handleSend() {
    if (!input.trim()) return
    const newMsg: Message = {
      id: Date.now(),
      original: input.trim(),
      result: mockTransliterate(input.trim()),
      model: activeModel.label,
    }
    setMessages(prev => [...prev, newMsg])
    setInput("")
  }

  // Auto-scroll to latest message
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
      boxShadow:
        "inset 0 2px 0 rgba(255, 255, 255, 0.1), inset 0 -1px 0 rgba(0, 0, 0, 0.2), 0 32px 80px rgba(0, 0, 0, 0.6), 0 8px 24px rgba(0, 0, 0, 0.4)",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
      zIndex: 10,
    }}>
      {/* Top shine */}
      <div style={{
        pointerEvents: "none",
        position: "absolute",
        inset: 0,
        borderRadius: "28px",
        background: "linear-gradient(135deg, rgba(38, 198, 176, 0.15) 0%, rgba(255, 255, 255, 0.02) 40%, transparent 65%)",
      }} />

      {/* ── Logo (top-left) ───────────────────────── */}
      <div style={{
        flexShrink: 0,
        padding: "20px 28px 16px",
        display: "flex",
        alignItems: "center",
        gap: 8,
        borderBottom: "1px solid rgba(38, 198, 176, 0.1)",
        position: "relative",
        zIndex: 30,
      }}>
        <span style={{
          fontSize: 20,
          color: "#26C6B0",
          filter: "drop-shadow(0 0 8px rgba(38, 198, 176, 0.7))",
          lineHeight: 1,
        }}>𑀱</span>
        <span style={{
          fontFamily: "'Geist Sans', sans-serif",
          fontSize: 17,
          fontWeight: 800,
          letterSpacing: -0.5,
          color: "#ffffff",
          textShadow: "0 0 20px rgba(38, 198, 176, 0.3)",
        }}>shabda</span>
      </div>

      {/* ── Scrollable chat area ──────────────────── */}
      <div style={{
        flex: 1,
        overflowY: "auto",
        position: "relative",
        zIndex: 20,
        display: "flex",
        flexDirection: "column",
      }}>
        {messages.length === 0 ? (
          /* Welcome state — centred */
          <div style={{
            flex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: "0 48px",
          }}>
            <div style={{ textAlign: "center" }}>
              <h1 style={{
                fontFamily: "'Geist Sans', sans-serif",
                fontSize: "clamp(28px, 4vw, 48px)",
                fontWeight: 700,
                color: "#ffffff",
                margin: 0,
                letterSpacing: -1,
                lineHeight: 1.15,
                textShadow: "0 2px 40px rgba(38, 198, 176, 0.25)",
              }}>
                Welcome to Shabda
              </h1>
              <p style={{
                fontFamily: "'Geist Sans', sans-serif",
                fontSize: "clamp(14px, 1.8vw, 20px)",
                fontWeight: 400,
                color: "rgba(38, 198, 176, 0.75)",
                margin: "12px 0 0",
                letterSpacing: 0.2,
              }}>
                Let's start writing shabdas.
              </p>
            </div>
          </div>
        ) : (
          /* Chat history */
          <div style={{
            flex: 1,
            padding: "24px 48px 12px",
            display: "flex",
            flexDirection: "column",
            gap: 28,
          }}>
            {messages.map((msg, i) => (
              <div
                key={msg.id}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                  opacity: 0,
                  animation: `fadeSlideIn 0.4s ease ${i === messages.length - 1 ? "0s" : "0s"} forwards`,
                }}
              >
                {/* User bubble — right aligned */}
                <div style={{ display: "flex", justifyContent: "flex-end" }}>
                  <div style={{
                    background: "rgba(38, 198, 176, 0.12)",
                    border: "1px solid rgba(38, 198, 176, 0.25)",
                    borderRadius: "18px 18px 4px 18px",
                    padding: "10px 16px",
                    maxWidth: "65%",
                    fontFamily: "'Geist Sans', sans-serif",
                    fontSize: 15,
                    color: "rgba(255,255,255,0.85)",
                    fontWeight: 500,
                    lineHeight: 1.5,
                  }}>
                    {msg.original}
                  </div>
                </div>

                {/* Result bubble — left aligned */}
                <div style={{ display: "flex", justifyContent: "flex-start", gap: 12, alignItems: "flex-start" }}>
                  {/* Model avatar */}
                  <div style={{
                    width: 32,
                    height: 32,
                    borderRadius: "50%",
                    background: "rgba(38, 198, 176, 0.15)",
                    border: "1px solid rgba(38, 198, 176, 0.35)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                    marginTop: 2,
                    fontSize: 14,
                    color: "#26C6B0",
                  }}>𑀱</div>

                  <div style={{ display: "flex", flexDirection: "column", gap: 4, maxWidth: "70%" }}>
                    <span style={{
                      fontFamily: "'Geist Mono', monospace",
                      fontSize: 10,
                      fontWeight: 600,
                      color: "rgba(38, 198, 176, 0.5)",
                      letterSpacing: 1.2,
                      textTransform: "uppercase",
                    }}>{msg.model}</span>
                    <div style={{
                      background: "rgba(0, 0, 0, 0.35)",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                      borderRadius: "4px 18px 18px 18px",
                      padding: "14px 20px",
                      fontFamily: "'Noto Sans Devanagari', 'Geist Sans', sans-serif",
                      fontSize: "clamp(20px, 2.5vw, 28px)",
                      fontWeight: 700,
                      color: "#ffffff",
                      lineHeight: 1.4,
                      textShadow: "0 0 30px rgba(38, 198, 176, 0.3)",
                    }}>
                      {msg.result}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            {/* Scroll anchor */}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* ── Bottom Input Bar ───────────────────────── */}
      <div style={{
        flexShrink: 0,
        position: "relative",
        zIndex: 20,
        padding: "0 20px 20px",
      }}>
        {/* Model selection popup */}
        {showModelMenu && (
          <div style={{
            position: "absolute",
            bottom: "calc(100% - 12px)",
            left: 20,
            zIndex: 100,
            background: "rgba(8, 14, 14, 0.95)",
            backdropFilter: "blur(24px)",
            border: "1px solid rgba(38, 198, 176, 0.3)",
            borderRadius: 16,
            padding: 8,
            minWidth: 260,
            boxShadow: "0 -16px 48px rgba(0, 0, 0, 0.7), 0 4px 16px rgba(0, 0, 0, 0.4)",
          }}>
            <div style={{
              padding: "8px 14px 10px",
              fontSize: 11,
              fontWeight: 700,
              color: "rgba(38, 198, 176, 0.7)",
              textTransform: "uppercase",
              letterSpacing: 1,
              borderBottom: "1px solid rgba(255, 255, 255, 0.07)",
              marginBottom: 4,
              fontFamily: "'Geist Mono', monospace",
            }}>
              Select Neural Architecture
            </div>
            {MODELS.map(m => (
              <button
                key={m.id}
                onClick={() => { setSelectedModel(m.id); setShowModelMenu(false) }}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  width: "100%",
                  padding: "10px 14px",
                  borderRadius: 10,
                  background: selectedModel === m.id ? "rgba(38, 198, 176, 0.12)" : "transparent",
                  border: selectedModel === m.id ? "1px solid rgba(38, 198, 176, 0.25)" : "1px solid transparent",
                  cursor: "pointer",
                  textAlign: "left",
                  transition: "all 0.15s",
                  marginBottom: 2,
                }}
              >
                <span style={{
                  fontSize: 13,
                  fontWeight: 600,
                  color: selectedModel === m.id ? "#26C6B0" : "#e2e8f0",
                  fontFamily: "'Geist Sans', sans-serif",
                }}>
                  {selectedModel === m.id && "✓ "}{m.label}
                </span>
                <span style={{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginTop: 2, fontFamily: "'Geist Sans', sans-serif" }}>
                  {m.desc}
                </span>
              </button>
            ))}
          </div>
        )}

        {/* Input pill */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          background: "rgba(0, 0, 0, 0.45)",
          border: "1px solid rgba(38, 198, 176, 0.3)",
          borderRadius: 999,
          padding: "6px",
          boxShadow: "inset 0 2px 6px rgba(0, 0, 0, 0.4), 0 4px 24px rgba(0, 0, 0, 0.3)",
        }}>
          {/* + Model selector */}
          <button
            onClick={() => setShowModelMenu(p => !p)}
            title={`Model: ${activeModel.label}`}
            style={{
              flexShrink: 0,
              width: 38, height: 38,
              borderRadius: "50%",
              background: showModelMenu ? "rgba(38, 198, 176, 0.25)" : "rgba(38, 198, 176, 0.1)",
              border: "1px solid rgba(38, 198, 176, 0.4)",
              color: "#26C6B0",
              fontSize: 22, fontWeight: 300, lineHeight: 1,
              cursor: "pointer",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all 0.2s ease",
            }}
          >+</button>

          {/* Text input */}
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value.slice(0, 250))}
            onKeyDown={e => { if (e.key === "Enter") handleSend() }}
            placeholder="Type Hinglish text here…"
            style={{
              flex: 1,
              background: "transparent",
              border: "none",
              outline: "none",
              color: "#ffffff",
              fontSize: 15,
              fontFamily: "'Geist Sans', sans-serif",
              fontWeight: 500,
              padding: "0 4px",
            }}
          />

          {/* Arrow send */}
          <button
            onClick={handleSend}
            title="Transliterate"
            style={{
              flexShrink: 0,
              width: 38, height: 38,
              borderRadius: "50%",
              background: input.trim() ? "linear-gradient(135deg, #00897B, #26C6B0)" : "rgba(255, 255, 255, 0.06)",
              border: input.trim() ? "1px solid rgba(38, 198, 176, 0.5)" : "1px solid rgba(255, 255, 255, 0.12)",
              color: input.trim() ? "#ffffff" : "rgba(255,255,255,0.25)",
              cursor: input.trim() ? "pointer" : "default",
              display: "flex", alignItems: "center", justifyContent: "center",
              transition: "all 0.2s ease",
              boxShadow: input.trim() ? "0 4px 16px rgba(0, 137, 123, 0.4)" : "none",
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="5" y1="12" x2="19" y2="12" />
              <polyline points="12 5 19 12 12 19" />
            </svg>
          </button>
        </div>
      </div>

      <style>{`
        input::placeholder { color: rgba(255,255,255,0.25); }
        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(16px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        div::-webkit-scrollbar { width: 4px; }
        div::-webkit-scrollbar-track { background: transparent; }
        div::-webkit-scrollbar-thumb { background: rgba(38, 198, 176, 0.2); border-radius: 2px; }
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
