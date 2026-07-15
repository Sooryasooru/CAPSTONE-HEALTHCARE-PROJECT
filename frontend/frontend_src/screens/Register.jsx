import { useState } from "react";
import { registerHospital } from "../api";

// Register screen — Option B. Wired to the backend: creates the account
// via /auth/register, then moves to login on success.

export default function Register({ nav }) {
  const [hospital, setHospital] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const pwMismatch = confirm.length > 0 && password !== confirm;
  const valid =
    hospital.trim() &&
    username.trim() &&
    password.length >= 6 &&
    password === confirm;

  async function submit() {
    if (!valid || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await registerHospital(hospital, username, password);
      if (res.success) {
        nav("login");
      } else {
        setError(res.message || "Registration failed.");
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
          <h1 className="brand-title">Clinical intelligence, unified.</h1>
          <p className="brand-copy">
            Set up your hospital in minutes. Upload your first guideline and
            HAIP builds a private analytics and knowledge platform around it.
          </p>
          <ul className="brand-points">
            <li>Analytics, forecasting, and guideline search in one place</li>
            <li>Your first document builds an isolated knowledge base</li>
            <li>Every automated decision shown for human review</li>
          </ul>
        </div>
      </aside>

      <section className="split-form dark-form">
        <div className="form-inner">
          <header className="form-head">
            <h2>Register your hospital</h2>
            <p>Create an admin account and add your first clinical guideline.</p>
          </header>

          <FloatField label="Hospital name" value={hospital} onChange={setHospital} placeholder="St. Mary's General" />
          <FloatField label="Admin username" value={username} onChange={setUsername} placeholder="username" />

          <div className="field-row">
            <FloatField label="Password" type="password" value={password} onChange={setPassword} placeholder="min. 6 characters" />
            <FloatField label="Confirm" type="password" value={confirm} onChange={setConfirm} placeholder="re-enter" error={pwMismatch} />
          </div>
          {pwMismatch && <div className="field-hint error-hint">Passwords don't match.</div>}

          <label className="file-drop dark-drop">
            <input type="file" accept=".pdf,.docx,.txt" onChange={(e) => setFile(e.target.files[0] || null)} hidden />
            <span className="float-label">First guideline (PDF, DOCX, or TXT)</span>
            <span className="file-drop-text">{file ? file.name : "Click to choose a document"}</span>
          </label>

          {error && <div className="auth-error">{error}</div>}

          <button className="ask-btn auth-submit" onClick={submit} disabled={!valid || busy}>
            {busy ? "Creating…" : "Create account"}
          </button>

          <div className="auth-alt">
            Already registered?{" "}
            <button className="link-btn" onClick={() => nav("login")}>Sign in</button>
          </div>
        </div>
      </section>
    </div>
  );
}

function FloatField({ label, value, onChange, placeholder, type = "text", error }) {
  return (
    <div className="float-field" data-error={error ? "true" : null}>
      <span className="float-label">{label}</span>
      <input className="float-input" type={type} value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} />
    </div>
  );
}
