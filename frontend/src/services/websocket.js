const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws";

let socket = null;
let onMessageCallback = null;

export function connectWebSocket(onMessage) {
  onMessageCallback = onMessage;
  socket = new WebSocket(WS_URL);

  socket.onopen = () => console.log("WebSocket connected");
  socket.onclose = () => console.log("WebSocket disconnected");
  socket.onerror = (e) => console.error("WebSocket error", e);

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (onMessageCallback) onMessageCallback(data);
    } catch (e) {
      console.error("Failed to parse WS message", e);
    }
  };
}

export function sendLandmarks(landmarks) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ landmarks }));
  }
}

export function disconnectWebSocket() {
  if (socket) socket.close();
}
