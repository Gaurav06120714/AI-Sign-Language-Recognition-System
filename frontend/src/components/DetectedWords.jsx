export default function DetectedWords({ words }) {
  if (words.length === 0) {
    return (
      <div className="flex items-center justify-center h-16 rounded-xl border border-white/10 bg-white/5">
        <p className="text-sm text-white/30">Detected words will appear here...</p>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap gap-2 min-h-[4rem] p-3 rounded-xl border border-white/10 bg-white/5">
      {words.map((word, i) => (
        <span
          key={i}
          className="px-3 py-1 rounded-full text-sm font-semibold bg-indigo-600/70 text-white border border-indigo-400/40"
        >
          {word}
        </span>
      ))}
    </div>
  );
}
