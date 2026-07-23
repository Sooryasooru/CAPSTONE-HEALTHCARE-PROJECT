// Hospital data card → full-screen embed of the merged Dash app (8060).
// Two tabs: Clinical Analytics (Synthea) and Data Intake & Prediction
// (upload a dataset, build a dashboard, download a PDF report).
export default function HospitalData({ nav, back }) {
  return (
    <div className="dash-fullscreen">
      <button className="dash-back" onClick={() => nav(back)}>
        ← Back
      </button>
      <iframe
        className="dash-fullscreen-frame"
        src="http://16.170.171.18:8060"
        title="Hospital data — Analytics & Data Intake"
      />
    </div>
  );
}
