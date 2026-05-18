# ollama_client.py
import time
import requests

SYSTEM_PROMPT = (
    "You are a friendly Hindi-language assistant. "
    "The user's messages will arrive in Devanagari script (transliterated from Hinglish). "
    "Always reply in clear, natural Devanagari Hindi. "
    "Keep your replies concise (2-4 sentences max). "
    "Do not switch to English or Roman script in your replies."
)

class OllamaChat:
    """
    Wraps the Ollama /api/chat endpoint with multi-turn conversation history.
    Uses gemma4:e2b by default. History is bounded to the last max_history
    exchanges to prevent context overflow.
    """

    def __init__(self, model="gemma4:e2b", system_prompt=SYSTEM_PROMPT, max_history=6):
        self.model = model
        self.max_history = max_history
        # Proper system role — Ollama/Gemma supports this correctly
        self.history = [{"role": "system", "content": system_prompt}]

    def chat(self, user_hindi):
        """
        Sends a Devanagari message and returns (reply_text, elapsed_ms).
        Returns an error string if Ollama is not reachable.
        """
        self.history.append({"role": "user", "content": user_hindi})

        t_start = time.perf_counter()
        try:
            r = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": self.history,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512,    # Large enough to accommodate both thinking and direct reply
                        "top_p": 0.9,
                        "num_ctx": 2048,       # standard context size
                    }
                },
                timeout=90
            )
            r.raise_for_status()
            reply = r.json()["message"]["content"].strip()
            # Fallback if model returns empty string
            if not reply:
                reply = "माफ़ करें, मुझे समझ नहीं आया। कृपया दोबारा पूछें।"
        except requests.exceptions.ConnectionError:
            self.history.pop()   # Don't store failed turn
            return "[Ollama is not running. Start it with: ollama serve]", 0
        except requests.exceptions.Timeout:
            self.history.pop()
            return "[Ollama timed out. The model may still be loading.]", 0
        except Exception as e:
            self.history.pop()
            return f"[Ollama error: {e}]", 0

        elapsed_ms = int((time.perf_counter() - t_start) * 1000)

        self.history.append({"role": "assistant", "content": reply})

        # Bound history: keep first 2 entries (system seed) + last N exchanges
        if len(self.history) > 2 + 2 * self.max_history:
            self.history = self.history[:2] + self.history[-(2 * self.max_history):]

        return reply, elapsed_ms

    def stream_response(self, user_hindi):
        """
        Streams a Devanagari reply token-by-token as a Python generator.
        Each yield is a raw text chunk from Ollama's streaming API.
        """
        self.history.append({"role": "user", "content": user_hindi})
        try:
            r = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": self.history,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512,
                        "top_p": 0.9,
                        "num_ctx": 2048,
                    }
                },
                stream=True,
                timeout=90
            )
            r.raise_for_status()

            full_reply = ""
            for line in r.iter_lines():
                if not line:
                    continue
                import json as _json
                try:
                    data = _json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    full_reply += chunk
                    yield chunk
                if data.get("done"):
                    break

            # Store completed reply in history
            if full_reply:
                self.history.append({"role": "assistant", "content": full_reply})
                if len(self.history) > 2 + 2 * self.max_history:
                    self.history = self.history[:2] + self.history[-(2 * self.max_history):]

        except requests.exceptions.ConnectionError:
            self.history.pop()
            yield "⚠️ Ollama is not running. Start it with: ollama serve"
        except requests.exceptions.Timeout:
            self.history.pop()
            yield "⚠️ Ollama timed out. The model may still be loading."
        except Exception as e:
            self.history.pop()
            yield f"⚠️ Ollama error: {e}"

    def reset(self):
        """Clears conversation history back to just the system prompt."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def is_alive(self):
        """Returns True if the Ollama server is reachable."""
        try:
            r = requests.get("http://localhost:11434/", timeout=2)
            return r.status_code == 200
        except Exception:
            return False
