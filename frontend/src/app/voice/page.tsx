"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, MicOff, Volume2, VolumeX, RotateCcw } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type Msg = { role: "user" | "ai"; text: string };
type Phase = "idle" | "listening" | "thinking" | "speaking";

const SUGGESTIONS = [
  "What should I stock next?",
  "Which category is growing fastest?",
  "What do customers buy together?",
  "How often do customers return?",
];

export default function VoiceAssistantPage() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [interim, setInterim] = useState("");
  const [typed, setTyped] = useState("");
  const [muted, setMuted] = useState(false);
  const [supported, setSupported] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const recRef = useRef<any>(null);
  const mutedRef = useRef(false);
  const endRef = useRef<HTMLDivElement>(null);

  mutedRef.current = muted;

  useEffect(() => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setSupported(false);
    }
    return () => {
      recRef.current?.stream?.getTracks().forEach((t: MediaStreamTrack) => t.stop());
      window.speechSynthesis?.cancel();
    };
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, interim]);

  const speak = (text: string) => {
    if (mutedRef.current || !window.speechSynthesis) {
      setPhase("idle");
      return;
    }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1;
    u.onstart = () => setPhase("speaking");
    u.onend = () => setPhase("idle");
    u.onerror = () => setPhase("idle");
    window.speechSynthesis.speak(u);
  };

  const ask = async (q: string) => {
    const question = q.trim();
    if (!question) return;
    setError(null);
    setMessages((m) => [...m, { role: "user", text: question }]);
    setPhase("thinking");
    try {
      const res = await fetch(`${API_BASE}/api/insights/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, conversation_id: "voice-session" }),
      });
      if (!res.ok) throw new Error("Assistant unavailable");
      const data = await res.json();
      const answer = data.answer || "I couldn't find an answer for that.";
      setMessages((m) => [...m, { role: "ai", text: answer }]);
      speak(answer);
    } catch {
      setError("Could not reach the assistant. Is the backend running?");
      setPhase("idle");
    }
  };

  const toggleMic = async () => {
    setError(null);
    if (phase === "listening") {
      recRef.current?.stop();
      return;
    }
    window.speechSynthesis?.cancel();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        if (!chunks.length) {
          setPhase("idle");
          return;
        }
        setPhase("thinking");
        const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
        const buf = new Uint8Array(await blob.arrayBuffer());
        let bin = "";
        for (let i = 0; i < buf.length; i++) bin += String.fromCharCode(buf[i]);
        try {
          const res = await fetch(`${API_BASE}/api/voice/transcribe`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio_base64: btoa(bin), mime_type: blob.type }),
          });
          if (!res.ok) throw new Error();
          const { text } = await res.json();
          if (!text) throw new Error();
          ask(text);
        } catch {
          setError("Couldn't understand the audio. Try again or type your question below.");
          setPhase("idle");
        }
      };
      recRef.current = rec;
      rec.start();
      setPhase("listening");
    } catch {
      setError("Microphone unavailable or permission denied. You can type your question below.");
      setPhase("idle");
    }
  };

  const reset = () => {
    window.speechSynthesis?.cancel();
    if (recRef.current?.state === "recording") recRef.current.stop();
    setMessages([]);
    setInterim("");
    setPhase("idle");
    setError(null);
  };

  const status: Record<Phase, string> = {
    idle: "Tap the mic and ask a question",
    listening: "Listening…",
    thinking: "Thinking…",
    speaking: "Speaking…",
  };

  return (
    <div className="p-6 md:p-8 max-w-3xl mx-auto flex flex-col min-h-[calc(100vh-6rem)]">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Voice Assistant</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Talk to your store data. Answers come from your real analytics, powered by Gemini.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => {
              if (!muted) window.speechSynthesis?.cancel();
              setMuted((v) => !v);
            }}
            title={muted ? "Unmute voice replies" : "Mute voice replies"}
            className="p-2 rounded-lg bg-secondary/40 hover:bg-secondary/70 text-muted-foreground hover:text-white"
          >
            {muted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
          </button>
          <button
            onClick={reset}
            title="Clear conversation"
            className="p-2 rounded-lg bg-secondary/40 hover:bg-secondary/70 text-muted-foreground hover:text-white"
          >
            <RotateCcw className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 my-6 space-y-3 overflow-y-auto">
        {messages.length === 0 && !interim && (
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground">Try asking:</p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => ask(s)}
                  className="px-3 py-1.5 rounded-full text-xs bg-secondary/40 hover:bg-indigo-500/20 border border-border/40 text-zinc-200 transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className={`max-w-[85%] px-3.5 py-2.5 rounded-2xl text-sm leading-relaxed ${
              m.role === "user" ? "ml-auto bg-indigo-500/20 text-white" : "bg-secondary/40 text-zinc-200"
            }`}
          >
            {m.text}
          </motion.div>
        ))}
        {interim && (
          <div className="ml-auto max-w-[85%] px-3.5 py-2.5 rounded-2xl text-sm bg-indigo-500/10 text-zinc-400 italic">
            {interim}
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="flex flex-col items-center gap-3 pb-2">
        {error && <p className="text-xs text-red-400">{error}</p>}
        {!supported && (
          <p className="text-xs text-amber-400">
            Audio recording isn&apos;t available in this browser. You can still type questions.
          </p>
        )}
        <div className="relative flex items-center justify-center h-24 w-24">
          <AnimatePresence>
            {(phase === "listening" || phase === "speaking") && (
              <motion.span
                key={phase}
                initial={{ scale: 1, opacity: 0.5 }}
                animate={{ scale: 1.7, opacity: 0 }}
                transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }}
                className={`absolute h-16 w-16 rounded-full ${
                  phase === "listening" ? "bg-indigo-500" : "bg-emerald-500"
                }`}
              />
            )}
          </AnimatePresence>
          <motion.button
            whileTap={{ scale: 0.93 }}
            onClick={toggleMic}
            disabled={!supported || phase === "thinking"}
            className={`relative h-16 w-16 rounded-full flex items-center justify-center text-white shadow-lg transition-colors disabled:opacity-50 ${
              phase === "listening" ? "bg-rose-500 hover:bg-rose-600" : "bg-indigo-500 hover:bg-indigo-600"
            }`}
            aria-label={phase === "listening" ? "Stop listening" : "Start listening"}
          >
            {phase === "listening" ? <MicOff className="h-6 w-6" /> : <Mic className="h-6 w-6" />}
          </motion.button>
        </div>
        <p className="text-xs text-muted-foreground">{status[phase]}</p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (typed.trim() && phase !== "thinking") {
              ask(typed);
              setTyped("");
            }
          }}
          className="w-full flex gap-2 mt-2"
        >
          <input
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            placeholder="…or type a question"
            className="flex-1 bg-background/80 border border-border/60 rounded-lg px-3 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
          <button
            type="submit"
            className="px-3 py-2 rounded-lg bg-indigo-500 hover:bg-indigo-600 text-white text-xs font-medium"
          >
            Ask
          </button>
        </form>
      </div>
    </div>
  );
}
