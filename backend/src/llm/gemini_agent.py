"""
Google AI Studio Gemini Text-Generation Agent
=============================================
Implements the official Gemini REST text-generation pattern
(system_instruction + contents + generationConfig) grounded in
the Intelligent Pattern Discovery Engine's structured retrieval context.
"""

import os
import json
import ssl
import urllib.request
from typing import Dict, Any, Optional


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()


def _load_google_api_key() -> Optional[str]:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        return key
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        ".env",
    )
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    if k.strip().lower() in ("google_api_key", "gemini_api_key"):
                        return v.strip().strip('"').strip("'")
    return None


class GeminiRetailAgent:
    SYSTEM_INSTRUCTION = (
        "You are a friendly, intelligent retail business assistant. "
        "Explain insights in simple, everyday language that anyone with zero technical "
        "or data-science knowledge can understand. Avoid jargon like WAPE, MAD, robust z-score, "
        "lift, or support — instead say 'accuracy', 'unusual spike', 'often bought together', etc. "
        "Use ONLY the facts provided in the context. Never invent numbers or causal claims."
    )

    def __init__(self, model: Optional[str] = None):
        self.api_key = _load_google_api_key()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-latest")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, question: str, context: Dict[str, Any]) -> str:
        if not self.api_key:
            raise RuntimeError("Google API key not configured.")

        payload = {
            "system_instruction": {"parts": [{"text": self.SYSTEM_INSTRUCTION}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                f"Store Analytics Context:\n{json.dumps(context, indent=2)[:4000]}\n\n"
                                f"User Question: {question}\n"
                                "Provide a clear, helpful answer in 2-4 plain-English sentences."
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300},
        }

        for model_id in [self.model, "gemini-flash-lite-latest"]:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self.api_key}"
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=20.0, context=_ssl_context()) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                continue

        raise RuntimeError("Gemini text-generation endpoints temporarily unavailable.")


def transcribe_audio(audio_b64: str, mime_type: str = "audio/webm") -> str:
    """Transcribes recorded speech with Gemini (inline audio input)."""
    api_key = _load_google_api_key()
    if not api_key:
        raise RuntimeError("Google API key not configured.")
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": "Transcribe this audio exactly. Return only the spoken words, nothing else."},
                {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
            ],
        }],
        "generationConfig": {"temperature": 0.0},
    }
    last_err = None
    for model_id in ["gemini-flash-latest", "gemini-flash-lite-latest"]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30.0, context=_ssl_context()) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Transcription unavailable: {last_err}")
