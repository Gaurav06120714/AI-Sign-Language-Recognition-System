import { useState, useEffect, useRef } from "react";
import CameraFeed from "./components/CameraFeed";
import GestureDisplay from "./components/GestureDisplay";
import DetectedWords from "./components/DetectedWords";
import SentenceOutput from "./components/SentenceOutput";
import { connectWebSocket, disconnectWebSocket } from "./services/websocket";

const BUFFER_THRESHOLD = 8; // consistent frames before accepting a word

export default function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [gesture, setGesture] = useState({ label: null, confidence: 0, accepted: false });
  const [words, setWords] = useState([]);
  const [sentence, setSentence] = useState("");

  const bufferRef = useRef({ label: null, count: 0 });

  useEffect(() => {
    connectWebSocket((data) => {
      if (data.error || !data.label) return;

      setGesture(data);

      if (!data.accepted) return;

      const buf = bufferRef.current;
      if (data.label === buf.label) {
        buf.count += 1;
        if (buf.count === BUFFER_THRESHOLD) {
          setWords((prev) => {
            if (prev[prev.length - 1] === data.label) return prev;
            const next = [...prev, data.label];
            setSentence(next.join(" "));
            return next;
          });
        }
      } else {
        bufferRef.current = { label: data.label, count: 1 };
      }
    });

    return () => disconnectWebSocket();
  }, []);

  const handleSpeak = () => {
    if (!sentence) return;
    const utter = new SpeechSynthesisUtterance(sentence);
    utter.rate = 0.9;
    window.speechSynthesis.speak(utter);
  };

  const handleClear = () => {
    setWords([]);
    setSentence("");
    bufferRef.current = { label: null, count: 0 };
  };

  return (
    <div className="min-h-screen bg-[#0f1117] text-white p-6">
      {/* Header */}
      <div className="max-w-5xl mx-auto mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-black text-white tracking-tight">
            🤟 AI Sign Language
          </h1>
          <p className="text-white/40 text-sm mt-1">
            Real-time hand gesture recognition powered by MediaPipe + FastAPI
          </p>
        </div>
        <button
          onClick={() => setIsRunning((v) => !v)}
          className={`px-5 py-2.5 rounded-xl font-semibold text-sm transition ${
            isRunning
              ? "bg-red-600 hover:bg-red-500"
              : "bg-indigo-600 hover:bg-indigo-500"
          }`}
        >
          {isRunning ? "⏸ Pause" : "▶ Start"}
        </button>
      </div>

      {/* Main Grid */}
      <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Camera */}
        <div className="space-y-4">
          <CameraFeed isRunning={isRunning} />

          {/* Status bar */}
          <div className="flex items-center gap-3 px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-xs text-white/40">
            <span className={`w-2 h-2 rounded-full ${isRunning ? "bg-green-400 animate-pulse" : "bg-gray-600"}`} />
            {isRunning ? "Sending landmarks to backend via WebSocket" : "Press Start to begin detection"}
          </div>
        </div>

        {/* Right panel */}
        <div className="space-y-4">
          <GestureDisplay
            gesture={gesture.label}
            confidence={gesture.confidence}
            accepted={gesture.accepted}
          />

          <div>
            <p className="text-xs uppercase tracking-widest text-white/40 font-semibold mb-2">
              Word History
            </p>
            <DetectedWords words={words} />
          </div>

          <SentenceOutput
            sentence={sentence}
            onClear={handleClear}
            onSpeak={handleSpeak}
          />
        </div>
      </div>

      {/* Footer */}
      <p className="text-center text-white/20 text-xs mt-10">
        MediaPipe JS runs in browser · Features sent via WebSocket · FastAPI predicts gesture
      </p>
    </div>
  );
}
