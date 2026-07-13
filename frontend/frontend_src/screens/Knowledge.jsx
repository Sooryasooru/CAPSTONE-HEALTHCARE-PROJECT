// Knowledge card → full-screen embed of the Streamlit RAG chat (8501).
// The logged-in hospital is passed via URL so the chat scopes to it —
// no re-upload, no hospital picker needed.
export default function Knowledge({ nav, back, session }) {
  const hospital = session?.hospital || "";
  const src = `http://localhost:8501/?hospital=${encodeURIComponent(hospital)}`;
  return (
    <div className="dash-fullscreen">
      <button className="dash-back" onClick={() => nav(back)}>
        ← Back
      </button>
      <iframe
        className="dash-fullscreen-frame"
        src={src}
        title="Knowledge — Guideline Assistant"
      />
    </div>
  );
}
