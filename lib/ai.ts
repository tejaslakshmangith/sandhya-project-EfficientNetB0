export interface PredictionResult {
  class: "safe" | "unsafe";
  confidence: number;
  error?: string;
}

export async function predictImage(file: File): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

  const response = await fetch(`${apiUrl}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`API error ${response.status}: ${err}`);
  }

  return response.json();
}
