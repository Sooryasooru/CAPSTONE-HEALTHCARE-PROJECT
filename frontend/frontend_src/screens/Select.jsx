// Select screen — Option C: navy sidebar + light card list.
// Echoes the auth split-screen layout so the section screen ties
// visually to the Register/Login pages (navy + teal).
const CARDS = [
  {
    id: "hospital",
    accent: "var(--teal)",
    tint: "rgba(15,181,166,0.15)",
    title: "Hospital data",
    desc: "Admissions, forecasts, and analytics.",
    icon: "M4 20V10M10 20V4M16 20v-7M20 20H2",
  },
  {
    id: "doctors",
    accent: "#4f8ff0",
    tint: "rgba(79,143,240,0.15)",
    title: "Doctors",
    desc: "Browse providers and their details.",
    icon: "M12 12a4 4 0 100-8 4 4 0 000 8zM4 21c0-4 4-6 8-6s8 2 8 6",
  },
  {
    id: "knowledge",
    accent: "#b07de0",
    tint: "rgba(176,125,224,0.15)",
    title: "Knowledge base",
    desc: "Ask questions of your clinical guidelines.",
    icon: "M4 5a2 2 0 012-2h13v16H6a2 2 0 00-2 2V5zM19 3v18",
  },
  {
    id: "askhaip",
    accent: "#f0a54f",
    tint: "rgba(240,165,79,0.15)",
    title: "Ask HAIP",
    desc: "Chat with the AI clinical agent.",
    icon: "M12 2a10 10 0 100 20 10 10 0 000-20zM8 10h8M8 14h5",
  },
];

export default function Select({ nav }) {
  return (
    <main className="haip-main select-c">
      <header className="select-c-head">
        <div className="select-c-mark">HAIP</div>
        <h2>Choose a section</h2>
        <p>Everything for your hospital, behind one door.</p>
      </header>

      <div className="select-c-list">
        {CARDS.map((c) => (
          <button
            key={c.id}
            className="c-row"
            style={{ "--accent": c.accent, "--tint": c.tint }}
            onClick={() => nav(c.id)}
          >
            <span className="c-row-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                   stroke="currentColor" strokeWidth="2"
                   strokeLinecap="round" strokeLinejoin="round">
                <path d={c.icon} />
              </svg>
            </span>
            <span className="c-row-text">
              <span className="c-row-title">{c.title}</span>
              <span className="c-row-desc">{c.desc}</span>
            </span>
          </button>
        ))}
      </div>
    </main>
  );
}
