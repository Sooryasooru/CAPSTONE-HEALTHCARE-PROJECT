import { useState } from "react";
import { askHaip } from "../api";

// Hospital data screen — ask a question, route it through the engines,
// show the routing decision alongside the answer (routing stays visible).
export default function HospitalData({ nav, back }) {
  const [q, setQ] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function ask() {
    if (!q.trim() || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const data = await askHaip(q);
      setResult(data);
    } catch (e) {
      setError("Can't reach the backend. Start it with: uvicorn api.main:app --port 8000");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="haip-main">
      <div className="placeholder">
        <h2>Hospital data</h2>
        <p>Ask about admissions, forecasts, and analytics.</p>

        <textarea
          className="ask-input"
          rows={3}
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. Forecast admissions for next week"
        />
        <button className="ask-btn" onClick={ask} disabled={busy}>
          {busy ? "Routing…" : "Ask"}
        </button>

        {error && <div className="auth-error">{error}</div>}

        {result && (
          <div className="route-result">
            <div className="route-decision">
              <strong>{result.decision.engine}</strong>
              {" · confidence "}
              {result.decision.confidence}
              <div className="route-reason">{result.decision.reason}</div>
            </div>
            <pre className="route-answer">
              {typeof result.answer === "string"
                ? result.answer
                : JSON.stringify(result.answer, null, 2)}
            </pre>
          </div>
        )}

        {/* TODO step 3: file upload -> ingest_pdf() */}

        {back && (
          <button className="ask-btn" onClick={() => nav(back)}>
            ← Back
          </button>
        )}
      </div>
    </main>
  );
}
