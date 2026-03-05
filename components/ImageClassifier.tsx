"use client";

import { useState, useRef, useCallback } from "react";
import { predictImage, PredictionResult } from "@/lib/ai";

type Status = "idle" | "loading" | "success" | "error";

export default function ImageClassifier() {
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
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
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setErrorMsg(msg);
      setStatus("error");
    }
  }, []);

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
            Make sure the FastAPI backend is running on{" "}
            <code className="text-gray-400">localhost:8000</code>
          </p>
          <button
            onClick={reset}
            className="text-sm text-gray-500 hover:text-gray-300 transition-colors underline underline-offset-2"
          >
            Try again
          </button>
        </div>
      )}
    </div>
  );
}
