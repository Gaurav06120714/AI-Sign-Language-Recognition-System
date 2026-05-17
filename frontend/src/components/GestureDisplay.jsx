export default function GestureDisplay({ gesture, confidence, accepted }) {
  const pct = Math.round((confidence || 0) * 100);
  const barColor = accepted ? "bg-green-500" : confidence > 0.4 ? "bg-yellow-500" : "bg-red-500";

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-widest text-white/40 font-semibold">Current Gesture</span>
        {accepted && (
          <span className="text-xs text-green-400 font-semibold">✓ Accepted</span>
        )}
      </div>

      <div className="flex items-end gap-4">
        <span className="text-6xl font-black text-indigo-400">
          {gesture || "—"}
        </span>
        <span className="text-2xl font-bold text-white/60 mb-1">{pct}%</span>
      </div>

      {/* Confidence bar */}
      <div className="w-full h-2 rounded-full bg-white/10">
        <div
          className={`h-2 rounded-full transition-all duration-150 ${barColor}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-white/25">
        <span>0%</span>
        <span className="text-white/40">Threshold 70%</span>
        <span>100%</span>
      </div>
    </div>
  );
}
