// Talks to the HAIP FastAPI backend.
// One function: send a question to /route, get back the routing decision
// plus the engine's answer.

const API_BASE = "http://127.0.0.1:8000";

export async function askHaip(question) {
  const res = await fetch(`${API_BASE}/route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  if (!res.ok) {
    throw new Error(`Backend responded ${res.status}`);
  }
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

// --- auth ---

export async function registerHospital(hospital_name, username, password) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hospital_name, username, password }),
  });
  if (!res.ok) throw new Error(`Backend responded ${res.status}`);
  return res.json(); // { success, message }
}

export async function loginHospital(username, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error(`Backend responded ${res.status}`);
  return res.json(); // { success, hospital, message? }
}
