export default function SentenceOutput({ sentence, onClear, onSpeak }) {
  const handleCopy = () => {
    if (sentence) navigator.clipboard.writeText(sentence);
  };

  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-widest text-white/40 font-semibold">Sentence</span>
        <div className="flex gap-2">
          <button
            onClick={handleCopy}
            disabled={!sentence}
            className="px-3 py-1 text-xs rounded-lg bg-white/10 hover:bg-white/20 disabled:opacity-30 transition"
          >
            Copy
          </button>
          <button
            onClick={onSpeak}
            disabled={!sentence}
            className="px-3 py-1 text-xs rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 transition"
          >
            🔊 Speak
          </button>
          <button
            onClick={onClear}
            className="px-3 py-1 text-xs rounded-lg bg-red-600/60 hover:bg-red-500/80 transition"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="min-h-[3rem] flex items-center">
        {sentence ? (
          <p className="text-2xl font-bold text-white tracking-wide">{sentence}</p>
        ) : (
          <p className="text-white/25 text-sm">Your sentence will appear here...</p>
        )}
      </div>
    </div>
  );
}
