export interface PredictionResult {
  class: "safe" | "unsafe";
  confidence: number;
  prediction_id?: number;
  session_id?: string;
  timestamp?: string;
  error?: string;
}

export interface HistoryItem {
  id: number;
  session_id: string;
  filename: string | null;
  class: "safe" | "unsafe";
  confidence: number;
  timestamp: string;
}

export interface Stats {
  total: number;
  safe: number;
  unsafe: number;
  avg_confidence: number;
  today: number;
}

function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return "";
  let sid = localStorage.getItem("smartmine_session_id");
  if (!sid) {
    sid = crypto.randomUUID();
    localStorage.setItem("smartmine_session_id", sid);
  }
  return sid;
}

const getFlaskUrl = () =>
  process.env.NEXT_PUBLIC_FLASK_URL ?? "http://localhost:5000";

export async function predictImage(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", getOrCreateSessionId());

  const response = await fetch(`${getFlaskUrl()}/api/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`API error ${response.status}: ${err}`);
  }

  return response.json();
}

export async function fetchHistory(limit = 20): Promise<HistoryItem[]> {
  const sid = getOrCreateSessionId();
  const url = `${getFlaskUrl()}/api/history?session_id=${encodeURIComponent(sid)}&limit=${limit}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`History fetch error ${response.status}`);
  }
  return response.json();
}

export async function fetchStats(): Promise<Stats> {
  const response = await fetch(`${getFlaskUrl()}/api/stats`);
  if (!response.ok) {
    throw new Error(`Stats fetch error ${response.status}`);
  }
  return response.json();
}

export async function deletePrediction(id: number): Promise<void> {
  const response = await fetch(`${getFlaskUrl()}/api/history/${id}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Delete error ${response.status}`);
  }
}
