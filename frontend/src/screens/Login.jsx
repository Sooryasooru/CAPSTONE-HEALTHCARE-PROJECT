import { useState } from "react";

// Login screen — Option B style, mirrors Register.

export default function Login({ nav, onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [file, setFile] = useState(null);

  const valid = username.trim() && password.length > 0;

  function submit() {
    if (!valid) return;
    onLogin({ hospital: username, username });
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
          <FloatField label="Password" type="password" value={password} onChange={setPassword} placeholder="password" />

          <label className="file-drop dark-drop">
            <input type="file" accept=".pdf,.docx,.txt" onChange={(e) => setFile(e.target.files[0] || null)} hidden />
            <span className="float-label">Add a guideline (optional)</span>
            <span className="file-drop-text">{file ? file.name : "Click to choose a document"}</span>
          </label>

          <button className="ask-btn auth-submit" onClick={submit} disabled={!valid}>
            Sign in
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

function FloatField({ label, value, onChange, placeholder, type = "text" }) {
  return (
    <div className="float-field">
      <span className="float-label">{label}</span>
      <input className="float-input" type={type} value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} />
    </div>
  );
}
