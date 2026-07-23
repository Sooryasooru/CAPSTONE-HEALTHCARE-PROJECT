// Doctors card → full-screen embed of the doctor Dash app (port 8051).
// CSV upload → department cards → drill into a department → per-doctor
// dashboards all live in the Dash app.
export default function Doctors({ nav, back }) {
  return (
    <div className="dash-fullscreen">
      <button className="dash-back" onClick={() => nav(back)}>
        ← Back
      </button>
      <iframe
        className="dash-fullscreen-frame"
        src="http://16.170.171.18:8051"
        title="Doctors — Department & Provider Analytics"
      />
    </div>
  );
}