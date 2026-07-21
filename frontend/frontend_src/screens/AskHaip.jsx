// Ask HAIP card → native chat with the LangGraph agent (8062).
// Unlike the other cards this is NOT an iframe: it's a React chat that
// POSTs to /agent/chat and renders the answer plus the `tools_used`
// reasoning trail, so the agent's work is visible to the clinician.
import { useState, useRef, useEffect } from "react";

const AGENT_URL = "http://localhost:8062/agent/chat";

const TOOL_LABELS = {
  search_guidelines: "Searched clinical guidelines",
  get_hospital_kpis: "Pulled hospital KPIs",
  forecast_admissions: "Forecast admissions",
  get_doctor_stats: "Read doctor stats",
};

const SUGGESTIONS = [
  "What is the sepsis screening protocol, and what are our current KPIs?",
  "How is delirium managed in hospital patients?",
  "Forecast next week's admissions and flag any risks.",
];

export default function AskHaip({ nav, back }) {
  const [messages, setMessages] = useState([]); // {role, text, tools?}
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = async (question) => {
    const q = (question ?? input).trim();
    if (!q || loading) return;

    setError("");
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);

    try {
      const res = await fetch(AGENT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          // Replay prior turns so the agent has multi-turn context.
          history: messages.map((m) => ({
            role: m.role === "user" ? "user" : "assistant",
            content: m.text,
          })),
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        // Backend sends a user-friendly `detail` (e.g. quota 503).
        throw new Error(data.detail || `Agent returned ${res.status}`);
      }
      setMessages((m) => [
        ...m,
        { role: "agent", text: data.answer, tools: data.tools_used || [] },
      ]);
    } catch (e) {
      setError(e.message || "Could not reach the agent.");
    } finally {
      setLoading(false);
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <main className="haip-main askhaip">
      <div className="askhaip-bar">
        <button className="dash-back" onClick={() => nav(back)}>
          ← Back
        </button>
        <div className="askhaip-title">
          <span className="askhaip-dot" />
          Ask HAIP · clinical agent
        </div>
      </div>

      <div className="askhaip-thread">
        {messages.length === 0 && (
          <div className="askhaip-empty">
            <h3>Ask across your whole hospital</h3>
            <p>
              The agent reasons over clinical guidelines, live KPIs, and
              admission forecasts — and shows every tool it used.
            </p>
            <div className="askhaip-suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="askhaip-chip" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`askhaip-msg ${m.role}`}>
            {m.role === "agent" && m.tools?.length > 0 && (
              <div className="askhaip-tools">
                {m.tools.map((t, j) => (
                  <span key={j} className="askhaip-tool">
                    <span className="askhaip-tool-tick" />
                    {TOOL_LABELS[t] || t}
                  </span>
                ))}
              </div>
            )}
            <div className="askhaip-bubble">{m.text}</div>
          </div>
        ))}

        {loading && (
          <div className="askhaip-msg agent">
            <div className="askhaip-bubble askhaip-thinking">
              <span /> <span /> <span />
            </div>
          </div>
        )}

        {error && <div className="askhaip-error">{error}</div>}
        <div ref={endRef} />
      </div>

      <div className="askhaip-composer">
        <textarea
          className="askhaip-input"
          rows={1}
          placeholder="Ask about guidelines, KPIs, or forecasts…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          disabled={loading}
        />
        <button
          className="askhaip-send"
          onClick={() => send()}
          disabled={loading || !input.trim()}
        >
          Send
        </button>
      </div>

      <p className="askhaip-disclaimer">
        Triage aid on synthetic data · not a diagnostic decision · review every
        output.
      </p>
    </main>
  );
}