import { useState } from "react";
import { loginHospital } from "../api";

// Login screen — Option B style. Wired to the real backend:
// verifies credentials via /auth/login and rejects bad passwords.

export default function Login({ nav, onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const valid = username.trim() && password.length > 0;

  async function submit() {
    if (!valid || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await loginHospital(username, password);
      if (res.success) {
        onLogin({ hospital: res.hospital, username });
      } else {
        setError(res.message || "Invalid username or password.");
      }
    } catch (e) {
      setError("Can't reach the backend. Start it with: uvicorn api.main:app --port 8000");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="split">
      <aside className="split-brand">
        <div className="brand-inner">
          <div className="brand-mark">HAIP</div>
          <h1 className="brand-title">Welcome back.</h1>
          <p className="brand-copy">
            Sign in to reach your hospital's analytics, forecasting, and
            guideline knowledge base. Add more clinical documents any time.
          </p>
          <ul className="brand-points">
            <li>Every automated decision shown for human review</li>
            <li>Your guidelines stay isolated to your hospital</li>
            <li>Proof of concept on synthetic data</li>
          </ul>
        </div>
      </aside>

      <section className="split-form dark-form">
        <div className="form-inner">
          <header className="form-head">
            <h2>Sign in</h2>
            <p>Access your hospital's intelligence platform.</p>
          </header>

          <FloatField label="Admin username" value={username} onChange={setUsername} placeholder="username" />
          <FloatField label="Password" type="password" value={password} onChange={setPassword} placeholder="password" onEnter={submit} />

          <label className="file-drop dark-drop">
            <input type="file" accept=".pdf,.docx,.txt" onChange={(e) => setFile(e.target.files[0] || null)} hidden />
            <span className="float-label">Add a guideline (optional)</span>
            <span className="file-drop-text">{file ? file.name : "Click to choose a document"}</span>
          </label>

          {error && <div className="auth-error">{error}</div>}

          <button className="ask-btn auth-submit" onClick={submit} disabled={!valid || busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>

          <div className="auth-alt">
            New hospital?{" "}
            <button className="link-btn" onClick={() => nav("register")}>Register</button>
          </div>
        </div>
      </section>
    </div>
  );
}

function FloatField({ label, value, onChange, placeholder, type = "text", onEnter }) {
  return (
    <div className="float-field">
      <span className="float-label">{label}</span>
      <input
        className="float-input"
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onEnter && onEnter()}
        placeholder={placeholder}
      />
    </div>
  );
}
