import { useState } from "react";

// Register screen — Option B: split layout, floating labels inside dark fields.

export default function Register({ nav }) {
  const [hospital, setHospital] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [file, setFile] = useState(null);

  const pwMismatch = confirm.length > 0 && password !== confirm;
  const valid =
    hospital.trim() &&
    username.trim() &&
    password.length >= 6 &&
    password === confirm;

  function submit() {
    if (!valid) return;
    nav("login");
  }

  return (
    <div className="split">
      <aside className="split-brand">
        <div className="brand-inner">
          <div className="brand-mark">HAIP</div>
          <h1 className="brand-title">Clinical intelligence, unified.</h1>
          <p className="brand-copy">
            Analytics, forecasting, and guideline search behind one secure door.
            Register your hospital to build an isolated knowledge base from your
            own clinical documents.
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

          <button className="ask-btn auth-submit" onClick={submit} disabled={!valid}>
            Create account
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
