import { useEffect, useRef } from "react";
import { Hands, HAND_CONNECTIONS } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { sendLandmarks } from "../services/websocket";

export default function CameraFeed({ isRunning }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const cameraRef = useRef(null);
  const handsRef = useRef(null);

  useEffect(() => {
    const hands = new Hands({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5,
    });

    hands.onResults((results) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Mirror the video frame onto canvas
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(results.image, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];

        // Draw hand skeleton
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
          color: "#6366f1",
          lineWidth: 2,
        });
        drawLandmarks(ctx, landmarks, {
          color: "#a5b4fc",
          lineWidth: 1,
          radius: 4,
        });

        // Send landmarks to FastAPI backend
        if (isRunning) {
          sendLandmarks(landmarks);
        }
      }
    });

    handsRef.current = hands;

    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        await hands.send({ image: videoRef.current });
      },
      width: 640,
      height: 480,
    });

    camera.start();
    cameraRef.current = camera;

    return () => {
      camera.stop();
      hands.close();
    };
  }, []);

  // Pause/resume landmark sending based on isRunning
  useEffect(() => {}, [isRunning]);

  return (
    <div className="relative w-full rounded-2xl overflow-hidden border border-indigo-500/30 bg-black shadow-xl shadow-indigo-500/10">
      <video ref={videoRef} className="hidden" playsInline />
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        className="w-full h-full object-cover"
      />
      <div className="absolute top-3 left-3 flex items-center gap-2">
        <span className={`w-2.5 h-2.5 rounded-full ${isRunning ? "bg-green-400 animate-pulse" : "bg-gray-500"}`} />
        <span className="text-xs text-white/70">{isRunning ? "Detecting" : "Paused"}</span>
      </div>
    </div>
  );
}
