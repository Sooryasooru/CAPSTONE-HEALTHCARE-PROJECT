// Talks to the HAIP FastAPI backend.
// One function: send a question to /route, get back the routing decision
// plus the engine's answer.

const API_BASE = "http://127.0.0.1:8000";

// --- JWT session token (in-memory; cleared on refresh/sign-out) ---
let authToken = null;

export function setToken(token) {
  authToken = token || null;
}

export function clearToken() {
  authToken = null;
}

export function getToken() {
  return authToken;
}

// Build headers, attaching the bearer token when we have one.
function authHeaders(extra = {}) {
  const h = { ...extra };
  if (authToken) h["Authorization"] = `Bearer ${authToken}`;
  return h;
}

export async function askHaip(question) {
  const res = await fetch(`${API_BASE}/route`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
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
  const data = await res.json(); // { success, hospital, role, access_token }
  if (data.success && data.access_token) setToken(data.access_token);
  return data;
}

export async function fetchDoctors() {
  const res = await fetch(`${API_BASE}/doctors`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Backend responded ${res.status}`);
  return res.json(); // { departments: {dept: [{name, specialty, encounters}]}, total }
}
