import ImageClassifier from "@/components/ImageClassifier";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-slate-900 flex flex-col items-center justify-center px-4 py-16">
      {/* Header */}
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/20 rounded-full px-4 py-1.5 mb-6 text-blue-400 text-sm font-medium">
          <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></span>
          EfficientNetB0 · ~5M params · ~93–95% accuracy
        </div>
        <h1 className="text-4xl md:text-5xl font-bold text-white mb-4 tracking-tight">
          SmartMine{" "}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
            Safety Classifier
          </span>
        </h1>
        <p className="text-gray-400 text-lg max-w-xl mx-auto">
          Upload a mining site image and let our AI instantly classify it as{" "}
          <span className="text-green-400 font-medium">safe</span> or{" "}
          <span className="text-red-400 font-medium">unsafe</span> using
          EfficientNetB0.
        </p>
      </div>

      {/* Classifier */}
      <ImageClassifier />

      {/* Architecture info */}
      <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-3xl text-sm text-gray-500">
        {[
          { icon: "🖼️", label: "Upload Image", desc: "Any JPG / PNG / WebP" },
          { icon: "🧠", label: "EfficientNetB0", desc: "PyTorch · Compound Scaling" },
          { icon: "⚡", label: "FastAPI Backend", desc: "REST · JSON response" },
        ].map((item) => (
          <div
            key={item.label}
            className="flex flex-col items-center gap-1 bg-gray-900/50 border border-gray-800 rounded-xl p-4 text-center"
          >
            <span className="text-2xl mb-1">{item.icon}</span>
            <span className="font-medium text-gray-300">{item.label}</span>
            <span>{item.desc}</span>
          </div>
        ))}
      </div>
    </main>
  );
}
