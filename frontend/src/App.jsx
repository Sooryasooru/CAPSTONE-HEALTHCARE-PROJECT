import { useState, useEffect } from "react";
import { askHaip, checkHealth } from "./api";
import "./App.css";

const EXAMPLES = [
  "How many admissions did we have?",
  "Forecast admissions for the next 6 months",
  "What is the treatment guideline for sepsis?",
];

const ENGINE_LABELS = {
  analytics: "Analytics",
  prediction: "Prediction",
  rag: "Knowledge base",
};

export default function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [online, setOnline] = useState(null);

  useEffect(() => {
    checkHealth()
      .then(() => setOnline(true))
      .catch(() => setOnline(false));
  }, []);

  async function submit(q) {
    const query = (q ?? question).trim();
    if (!query) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await askHaip(query);
      setResult(data);
    } catch (e) {
      setError(
        "Can't reach the HAIP backend. Start it with: uvicorn api.main:app --port 8000"
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="haip">
      <header className="haip-header">
        <div className="mark">
          <span className="mark-dot" data-online={online} />
          HAIP
        </div>
        <div className="tagline">
          Healthcare Analytics &amp; Intelligence Platform
          <span className="poc">proof of concept · synthetic data</span>
        </div>
      </header>

      <main className="haip-main">
        <section className="ask">
          <label className="ask-label">Ask a question</label>
          <div className="ask-row">
            <input
              className="ask-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
              placeholder="e.g. forecast admissions for the next 6 months"
            />
            <button
              className="ask-btn"
              onClick={() => submit()}
              disabled={loading || !question.trim()}
            >
              {loading ? "Routing…" : "Ask"}
            </button>
          </div>
          <div className="examples">
            {EXAMPLES.map((ex) => (
              <button key={ex} className="chip" onClick={() => { setQuestion(ex); submit(ex); }}>
                {ex}
              </button>
            ))}
          </div>
        </section>

        {error && <div className="error">{error}</div>}

        {result && (
          <section className="result">
            <RoutingTrace decision={result.decision} />
            <Answer answer={result.answer} />
          </section>
        )}
      </main>

      <footer className="haip-footer">
        Outputs are triage aids on synthetic data, not diagnostic decisions.
        Every automated route is shown for human review.
      </footer>
    </div>
  );
}

function RoutingTrace({ decision }) {
  const pct = Math.round((decision.confidence || 0) * 100);
  return (
    <div className="trace">
      <div className="trace-head">
        <span className="trace-title">Routing trace</span>
        <span className="trace-engine">{ENGINE_LABELS[decision.engine] || decision.engine}</span>
      </div>
      <div className="trace-bar">
        <div className="trace-fill" style={{ width: `${pct}%` }} data-engine={decision.engine} />
        <span className="trace-pct">{pct}% confidence</span>
      </div>
      <div className="trace-reason">{decision.reason}</div>
      {decision.matched && decision.matched.length > 0 && (
        <div className="trace-kw">
          {decision.matched.map((k) => (
            <span key={k} className="kw">{k}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function Answer({ answer }) {
  if (!answer) return null;

  if (answer.answer_type === "admissions_summary") {
    const classes = Object.entries(answer.by_encounter_class || {});
    const max = Math.max(...classes.map(([, v]) => v), 1);
    return (
      <div className="answer">
        <div className="metric">
          <span className="metric-num">{answer.total_admissions_all_time.toLocaleString()}</span>
          <span className="metric-label">total admissions (all time)</span>
        </div>
        <div className="bars">
          {classes.map(([name, val]) => (
            <div className="bars-row" key={name}>
              <span className="bars-name">{name}</span>
              <div className="bars-track">
                <div className="bars-fill" style={{ width: `${(val / max) * 100}%` }} />
              </div>
              <span className="bars-val">{val.toLocaleString()}</span>
            </div>
          ))}
        </div>
        <div className="note">{answer.note}</div>
      </div>
    );
  }

  if (answer.answer_type === "admissions_forecast") {
    const max = Math.max(...answer.forecast.map((f) => f.forecast), 1);
    return (
      <div className="answer">
        <div className="answer-title">
          {answer.horizon_months}-month forecast
          <span className="method">{answer.method}</span>
        </div>
        <div className="forecast">
          {answer.forecast.map((f) => (
            <div className="fc-col" key={f.month}>
              <div className="fc-bar-wrap">
                <div className="fc-bar" style={{ height: `${(f.forecast / max) * 100}%` }} />
              </div>
              <span className="fc-val">{f.forecast}</span>
              <span className="fc-month">{f.month.slice(5)}</span>
            </div>
          ))}
        </div>
        <div className="note">{answer.note}</div>
      </div>
    );
  }

  if (answer.answer_type === "grounded_answer") {
    return (
      <div className="answer">
        <p className="grounded">{answer.answer}</p>
        {answer.citations && answer.citations.length > 0 && (
          <div className="citations">
            <span className="cite-label">Sources</span>
            <ol>
              {answer.citations.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ol>
          </div>
        )}
        <div className="note">
          {answer.note}
          {answer.latency_seconds != null && ` · ${answer.latency_seconds}s`}
        </div>
      </div>
    );
  }

  // unavailable / error / not_implemented
  return (
    <div className="answer">
      <div className="answer-fallback">{answer.note || "No answer available."}</div>
    </div>
  );
}
