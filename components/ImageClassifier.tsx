"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import {
  predictImage,
  fetchHistory,
  fetchStats,
  deletePrediction,
  PredictionResult,
  HistoryItem,
  Stats,
} from "@/lib/ai";

type Status = "idle" | "loading" | "success" | "error";

function relativeTime(iso: string): string {
  const diff = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

export default function ImageClassifier() {
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string>("");
  const [deleteError, setDeleteError] = useState<string>("");
  const [stats, setStats] = useState<Stats | null>(null);

  const refreshHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const [h, s] = await Promise.all([fetchHistory(20), fetchStats()]);
      setHistory(h);
      setStats(s);
    } catch {
      setHistoryError("History unavailable — make sure the Flask backend is running on localhost:5000.");
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("image/")) {
        setErrorMsg("Please upload a valid image file (JPG, PNG, WebP).");
        setStatus("error");
        return;
      }

      // Show preview
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(file);

      // Run inference
      setStatus("loading");
      setResult(null);
      setErrorMsg("");

      try {
        const prediction = await predictImage(file);
        setResult(prediction);
        setStatus("success");
        await refreshHistory();
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setErrorMsg(msg);
        setStatus("error");
      }
    },
    [refreshHistory]
  );

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const reset = () => {
    setPreview(null);
    setResult(null);
    setStatus("idle");
    setErrorMsg("");
    if (inputRef.current) inputRef.current.value = "";
  };

  const handleDelete = async (id: number) => {
    setDeleteError("");
    try {
      await deletePrediction(id);
      await refreshHistory();
    } catch {
      setDeleteError("Failed to delete prediction. Please try again.");
    }
  };

  const isSafe = result?.class === "safe";

  return (
    <div className="w-full max-w-lg space-y-6">
      {/* Drop Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => inputRef.current?.click()}
        className={`
          relative cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-200
          flex flex-col items-center justify-center gap-3 p-10
          ${isDragging
            ? "border-blue-400 bg-blue-500/10 scale-[1.01]"
            : "border-gray-700 hover:border-gray-500 bg-gray-900/40 hover:bg-gray-900/70"
          }
          ${status === "loading" ? "pointer-events-none opacity-60" : ""}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleInputChange}
        />

        {preview ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={preview}
            alt="Preview"
            className="max-h-56 rounded-xl object-contain"
          />
        ) : (
          <>
            <div className="text-4xl">📷</div>
            <div className="text-center">
              <p className="text-gray-300 font-medium">
                Drop an image here or click to browse
              </p>
              <p className="text-gray-500 text-sm mt-1">
                JPG, PNG, WebP supported
              </p>
            </div>
          </>
        )}
      </div>

      {/* Loading */}
      {status === "loading" && (
        <div className="flex items-center justify-center gap-3 text-blue-400 py-4">
          <svg
            className="animate-spin w-5 h-5"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8v8H4z"
            />
          </svg>
          <span className="font-medium">Running EfficientNetB0 inference…</span>
        </div>
      )}

      {/* Result */}
      {status === "success" && result && (
        <div
          className={`rounded-2xl border p-6 flex flex-col items-center gap-3 transition-all
            ${isSafe
              ? "border-green-500/40 bg-green-500/10"
              : "border-red-500/40 bg-red-500/10"
            }
          `}
        >
          <span className="text-5xl">{isSafe ? "✅" : "⚠️"}</span>
          <p
            className={`text-2xl font-bold tracking-wide uppercase ${
              isSafe ? "text-green-400" : "text-red-400"
            }`}
          >
            {result.class}
          </p>
          <div className="w-full bg-gray-800 rounded-full h-2.5 overflow-hidden">
            <div
              className={`h-2.5 rounded-full transition-all duration-700 ${
                isSafe ? "bg-green-400" : "bg-red-400"
              }`}
              style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
            />
          </div>
          <p className="text-gray-400 text-sm">
            Confidence:{" "}
            <span className="text-white font-semibold">
              {(result.confidence * 100).toFixed(1)}%
            </span>
          </p>
          <button
            onClick={reset}
            className="mt-2 text-sm text-gray-500 hover:text-gray-300 transition-colors underline underline-offset-2"
          >
            Classify another image
          </button>
        </div>
      )}

      {/* Error */}
      {status === "error" && (
        <div className="rounded-2xl border border-orange-500/40 bg-orange-500/10 p-5 text-center space-y-2">
          <p className="text-orange-400 font-semibold">⚠️ Error</p>
          <p className="text-gray-400 text-sm">{errorMsg}</p>
          <p className="text-gray-500 text-xs">
            Make sure the Flask backend is running on{" "}
            <code className="text-gray-400">localhost:5000</code>
          </p>
          <button
            onClick={reset}
            className="text-sm text-gray-500 hover:text-gray-300 transition-colors underline underline-offset-2"
          >
            Try again
          </button>
        </div>
      )}

      {/* Stats Bar */}
      {stats && (
        <div className="rounded-xl bg-gray-900/60 border border-gray-800 px-4 py-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-400">
          <span>Total: <span className="text-white font-semibold">{stats.total}</span></span>
          <span>Safe: <span className="text-green-400 font-semibold">{stats.safe}</span></span>
          <span>Unsafe: <span className="text-red-400 font-semibold">{stats.unsafe}</span></span>
          <span>Avg confidence: <span className="text-white font-semibold">{(stats.avg_confidence * 100).toFixed(1)}%</span></span>
          <span>Today: <span className="text-blue-400 font-semibold">{stats.today}</span></span>
        </div>
      )}

      {/* History Panel */}
      <div className="rounded-2xl border border-gray-800 bg-gray-900/40">
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h2 className="text-sm font-semibold text-gray-300">
            History
            {history.length > 0 && (
              <span className="ml-2 text-xs text-gray-500 font-normal">({history.length})</span>
            )}
          </h2>
          <button
            onClick={refreshHistory}
            className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
            aria-label="Refresh history"
          >
            ↻ Refresh
          </button>
        </div>

        {historyLoading ? (
          <div className="py-6 text-center text-gray-500 text-sm">Loading…</div>
        ) : historyError ? (
          <div className="py-4 px-4 text-center text-orange-400 text-xs">{historyError}</div>
        ) : history.length === 0 ? (
          <div className="py-6 text-center text-gray-600 text-sm">No predictions yet</div>
        ) : (
          <ul className="divide-y divide-gray-800 max-h-72 overflow-y-auto">
            {deleteError && (
              <li className="px-4 py-2 text-xs text-red-400">{deleteError}</li>
            )}
            {history.map((item) => (
              <li key={item.id} className="flex items-center gap-3 px-4 py-2.5">
                <span className="text-lg shrink-0">{item.class === "safe" ? "🟢" : "🔴"}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-300 truncate">{item.filename || "unknown"}</p>
                  <p className="text-xs text-gray-500">
                    {(item.confidence * 100).toFixed(1)}% · {relativeTime(item.timestamp)}
                  </p>
                </div>
                <button
                  onClick={() => handleDelete(item.id)}
                  className="text-gray-600 hover:text-red-400 transition-colors text-sm leading-none shrink-0"
                  aria-label="Delete prediction"
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
