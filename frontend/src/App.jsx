import { useState } from "react";
import Register from "./screens/Register";
import Login from "./screens/Login";
import Select from "./screens/Select";
import HospitalData from "./screens/HospitalData";
import "./App.css";

// The whole product flow is driven by one piece of state: `screen`.
// No router library — just conditional rendering. Keeps it light.
//
//   register -> login -> select -> (hospital | doctors | knowledge)
//
// Auth screens (register/login) use a full split-screen layout with their
// own brand panel, so the app header/footer are hidden on those.

const AUTH_SCREENS = ["register", "login"];

export default function App() {
  const [screen, setScreen] = useState("register");
  const [session, setSession] = useState(null); // { hospital, username }

  const nav = (to) => setScreen(to);
  const login = (info) => {
    setSession(info);
    setScreen("select");
  };
  const logout = () => {
    setSession(null);
    setScreen("register");
  };

  const isAuth = AUTH_SCREENS.includes(screen);

  return (
    <div className="haip">
      {!isAuth && <Header session={session} onLogout={logout} />}

      {screen === "register" && <Register nav={nav} />}
      {screen === "login" && <Login nav={nav} onLogin={login} />}
      {screen === "select" && <Select nav={nav} />}
      {screen === "hospital" && <HospitalData nav={nav} back="select" />}
      {screen === "doctors" && <Placeholder title="Doctors" nav={nav} back="select" />}
      {screen === "knowledge" && <Placeholder title="Knowledge base" nav={nav} back="select" />}

      {!isAuth && (
        <footer className="haip-footer">
          Outputs are triage aids on synthetic data, not diagnostic decisions.
          Every automated route is shown for human review.
        </footer>
      )}
    </div>
  );
}

function Header({ session, onLogout }) {
  return (
    <header className="haip-header">
      <div className="mark">
        <span className="mark-dot" data-online={session ? "true" : null} />
        HAIP
      </div>
      <div className="tagline">
        Healthcare Analytics &amp; Intelligence Platform
        <span className="poc">proof of concept · synthetic data</span>
      </div>
      {session && (
        <button className="header-logout" onClick={onLogout}>
          {session.hospital} · sign out
        </button>
      )}
    </header>
  );
}

function Placeholder({ title, nav, back }) {
  return (
    <main className="haip-main">
      <div className="placeholder">
        <h2>{title}</h2>
        <p>This screen is built in a later phase.</p>
        {back && (
          <button className="ask-btn" onClick={() => nav(back)}>
            ← Back
          </button>
        )}
      </div>
    </main>
  );
}
